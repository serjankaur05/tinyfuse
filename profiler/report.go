package main

import (
	"fmt"
	"html/template"
	"io"
)

type barRow struct {
	Label string
	Value float64
	Width float64 // percent of chart width
	Text  string
}

type kernelRow struct {
	Stat
	GFLOPS, GBps, Intensity float64
	PctPeak                 string
	Bound                   string
}

type cmpRow struct {
	Speedup
	Factor float64
}

type reportData struct {
	Trace       string
	Meta        map[string]interface{}
	Totals      []barRow
	Kernels     []kernelRow
	Compare     []cmpRow
	CmpA, CmpB  string
	ChartHeight int
}

const reportTmpl = `<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>kprof report</title>
<style>
:root{--bg:#fbfbfa;--fg:#1d1d1b;--muted:#6b6b66;--line:#e3e2dc;--bar:#2f6f8f;--bar2:#c2703d}
@media (prefers-color-scheme:dark){:root{--bg:#161614;--fg:#ecebe6;--muted:#9d9c95;--line:#33322e;--bar:#6fb3d2;--bar2:#e39a66}}
body{background:var(--bg);color:var(--fg);font:14px/1.5 ui-sans-serif,system-ui,sans-serif;margin:0;padding:24px 16px;max-width:1100px;margin-inline:auto}
h1{font-size:20px;margin:0 0 4px}h2{font-size:15px;margin:28px 0 8px}
.muted{color:var(--muted)}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:5px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}
.scroll{overflow-x:auto}svg text{fill:var(--fg);font-size:12px}
</style></head><body>
<h1>kprof report</h1>
<div class="muted">{{.Trace}}{{range $k, $v := .Meta}} &middot; {{$k}}={{$v}}{{end}}</div>

<h2>One forward pass, sum of median kernel times</h2>
<svg width="820" height="{{.ChartHeight}}" style="max-width:100%" role="img" aria-label="Time per variant">
{{range $i, $b := .Totals}}
<text x="0" y="{{rowY $i}}" dy="14">{{$b.Label}}</text>
<rect x="190" y="{{rowY $i}}" width="{{barPx $b.Width}}" height="18" fill="var(--bar)"></rect>
<text x="{{labelPx $b.Width}}" y="{{rowY $i}}" dy="14">{{$b.Text}}</text>
{{end}}
</svg>

{{if .Compare}}
<h2>{{.CmpB}} vs {{.CmpA}}, by layer</h2>
<div class="scroll"><table><tr><th>layer</th><th>kernel</th><th>{{.CmpA}} (us)</th><th>{{.CmpB}} (us)</th><th>speedup</th></tr>
{{range .Compare}}<tr><td>L{{printf "%.0f" .Layer}}</td><td>{{.Kernel}}</td><td>{{printf "%.1f" .BaseUS}}</td><td>{{printf "%.1f" .TestUS}}</td><td>{{printf "%.1fx" .Factor}}</td></tr>
{{end}}</table></div>
{{end}}

<h2>Kernels</h2>
<div class="scroll"><table>
<tr><th>variant</th><th>kernel</th><th>n</th><th>median us</th><th>p95 us</th><th>GFLOP/s</th><th>GB/s</th><th>FLOP/B</th><th>% peak</th><th>bound</th></tr>
{{range .Kernels}}<tr><td>{{.Variant}}</td><td>{{.Kernel}}</td><td>{{.Count}}</td><td>{{printf "%.1f" .Median}}</td><td>{{printf "%.1f" .P95}}</td><td>{{printf "%.2f" .GFLOPS}}</td><td>{{printf "%.2f" .GBps}}</td><td>{{printf "%.1f" .Intensity}}</td><td>{{.PctPeak}}</td><td>{{.Bound}}</td></tr>
{{end}}</table></div>
</body></html>
`

func writeHTML(w io.Writer, opt options, t *Trace, stats []Stat, order []string,
	totals map[string]float64, cmp []Speedup, a, b string) error {
	const chartPx = 560.0 // drawable bar width in px (viewBox-free, so keep modest)
	maxT := 0.0
	for _, v := range totals {
		if v > maxT {
			maxT = v
		}
	}
	d := reportData{Trace: opt.tracePath, Meta: t.Metadata, CmpA: a, CmpB: b}
	for _, n := range order {
		d.Totals = append(d.Totals, barRow{Label: n, Value: totals[n],
			Width: 100 * totals[n] / maxT, Text: fmt.Sprintf("%.0f us", totals[n])})
	}
	d.ChartHeight = 26*len(d.Totals) + 6
	for _, s := range stats {
		roof := Roofline{PeakGFLOPS: peakFor(t, s.Variant, opt.peak), BWGBps: opt.bw}
		pct := "-"
		if roof.PeakGFLOPS > 0 && s.FLOPs > 0 {
			pct = fmt.Sprintf("%.1f", 100*s.GFLOPS()/roof.PeakGFLOPS)
		}
		d.Kernels = append(d.Kernels, kernelRow{Stat: s, GFLOPS: s.GFLOPS(), GBps: s.GBps(),
			Intensity: s.Intensity(), PctPeak: pct, Bound: roof.Bound(s.Intensity())})
	}
	for _, c := range cmp {
		d.Compare = append(d.Compare, cmpRow{Speedup: c, Factor: c.BaseUS / c.TestUS})
	}
	funcs := template.FuncMap{
		"rowY":    func(i int) int { return 26 * i },
		"barPx":   func(pct float64) string { return fmt.Sprintf("%.0f", pct/100*chartPx) },
		"labelPx": func(pct float64) string { return fmt.Sprintf("%.0f", 196+pct/100*chartPx) },
	}
	tmpl, err := template.New("report").Funcs(funcs).Parse(reportTmpl)
	if err != nil {
		return err
	}
	return tmpl.Execute(w, d)
}
