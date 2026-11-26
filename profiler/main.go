// kprof: summarize kernel traces from tinyfuse benchmarks.
//
//	kprof -trace results/kernel_trace.json [-compare naive,simd_fused_omp]
//	      [-bw 20] [-peak 480] [-top 10] [-html report.html]
//
// For every (variant, kernel) it reports launches, median and p95 latency,
// achieved GFLOP/s and GB/s, arithmetic intensity, % of the machine's
// measured FMA peak, and whether the kernel sits on the compute or memory
// side of the roofline.
package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"sort"
	"strings"
	"text/tabwriter"
)

type options struct {
	tracePath string
	compare   string
	peak      float64
	bw        float64
	top       int
	html      string
}

func peakFor(t *Trace, variant string, override float64) float64 {
	if override > 0 {
		return override
	}
	if strings.Contains(variant, "omp") {
		if v := t.MetaFloat("peak_gflops_mt"); v > 0 {
			return v
		}
	}
	return t.MetaFloat("peak_gflops_1t")
}

func run(opt options, w io.Writer) error {
	f, err := os.Open(opt.tracePath)
	if err != nil {
		return err
	}
	defer f.Close()
	t, err := ParseTrace(f)
	if err != nil {
		return err
	}
	if len(t.Events) == 0 {
		return fmt.Errorf("%s: no complete (ph=X) events", opt.tracePath)
	}
	stats := Aggregate(t.Events)
	if opt.bw == 0 {
		opt.bw = t.MetaFloat("bw_gbps") // measured by bench, if present
	}

	fmt.Fprintf(w, "kprof | %s | %d events, %d kernels", opt.tracePath, len(t.Events), len(stats))
	if isa, ok := t.Metadata["isa"].(string); ok {
		fmt.Fprintf(w, " | isa=%s", isa)
	}
	if p := t.MetaFloat("peak_gflops_1t"); p > 0 {
		fmt.Fprintf(w, " | FMA peak %.0f GFLOP/s (1t), %.0f (all threads)", p, t.MetaFloat("peak_gflops_mt"))
	}
	fmt.Fprintln(w)
	fmt.Fprintln(w)

	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', tabwriter.AlignRight)
	fmt.Fprint(tw, "variant\tkernel\tn\tmedian_us\tp95_us\tGFLOP/s\tGB/s\tFLOP/B\tpct_peak\tbound\t\n")
	for _, s := range stats {
		roof := Roofline{PeakGFLOPS: peakFor(t, s.Variant, opt.peak), BWGBps: opt.bw}
		pct := "-"
		if roof.PeakGFLOPS > 0 && s.FLOPs > 0 {
			pct = fmt.Sprintf("%.1f", 100*s.GFLOPS()/roof.PeakGFLOPS)
		}
		fmt.Fprintf(tw, "%s\t%s\t%d\t%.1f\t%.1f\t%.2f\t%.2f\t%.1f\t%s\t%s\t\n",
			s.Variant, s.Kernel, s.Count, s.Median, s.P95, s.GFLOPS(), s.GBps(),
			s.Intensity(), pct, roof.Bound(s.Intensity()))
	}
	tw.Flush()

	totals := VariantTotals(stats)
	names := make([]string, 0, len(totals))
	for k := range totals {
		names = append(names, k)
	}
	sort.Slice(names, func(i, j int) bool { return totals[names[i]] > totals[names[j]] })
	fmt.Fprintln(w, "\nsum of median kernel times per variant (one pass):")
	slowest := totals[names[0]]
	for _, n := range names {
		fmt.Fprintf(w, "  %-24s %10.1f us   %6.1fx vs slowest\n", n, totals[n], slowest/totals[n])
	}

	if opt.top > 0 {
		hot := append([]Stat(nil), stats...)
		sort.Slice(hot, func(i, j int) bool { return hot[i].Total > hot[j].Total })
		grand := 0.0
		for _, s := range hot {
			grand += s.Total
		}
		if opt.top < len(hot) {
			hot = hot[:opt.top]
		}
		fmt.Fprintf(w, "\ntop %d hotspots by total time:\n", len(hot))
		for _, s := range hot {
			fmt.Fprintf(w, "  %5.1f%%  %-18s %s\n", 100*s.Total/grand, s.Variant, s.Kernel)
		}
	}

	var cmp []Speedup
	var cmpA, cmpB string
	if opt.compare != "" {
		parts := strings.SplitN(opt.compare, ",", 2)
		if len(parts) != 2 {
			return fmt.Errorf("-compare wants two variants: base,test")
		}
		cmpA, cmpB = parts[0], parts[1]
		cmp = Compare(stats, cmpA, cmpB)
		fmt.Fprintf(w, "\n%s vs %s (by layer):\n", cmpB, cmpA)
		for _, c := range cmp {
			fmt.Fprintf(w, "  L%-3.0f %-28s %10.1f -> %8.1f us   %6.1fx\n",
				c.Layer, c.Kernel, c.BaseUS, c.TestUS, c.BaseUS/c.TestUS)
		}
	}

	if opt.html != "" {
		hf, err := os.Create(opt.html)
		if err != nil {
			return err
		}
		defer hf.Close()
		if err := writeHTML(hf, opt, t, stats, names, totals, cmp, cmpA, cmpB); err != nil {
			return err
		}
		fmt.Fprintf(w, "\nwrote %s\n", opt.html)
	}
	return nil
}

func main() {
	var opt options
	flag.StringVar(&opt.tracePath, "trace", "", "Chrome/Perfetto trace JSON (required)")
	flag.StringVar(&opt.compare, "compare", "", "base,test variants to compare layer by layer")
	flag.Float64Var(&opt.peak, "peak", 0, "compute ceiling in GFLOP/s (default: measured peak in trace metadata)")
	flag.Float64Var(&opt.bw, "bw", 0, "memory bandwidth ceiling in GB/s (default: measured bandwidth in trace metadata)")
	flag.IntVar(&opt.top, "top", 5, "show the N kernels with the most total time")
	flag.StringVar(&opt.html, "html", "", "also write an HTML report to this path")
	flag.Parse()
	if opt.tracePath == "" {
		flag.Usage()
		os.Exit(2)
	}
	if err := run(opt, os.Stdout); err != nil {
		fmt.Fprintln(os.Stderr, "kprof:", err)
		os.Exit(1)
	}
}
