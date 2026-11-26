// Package main implements kprof, a small kernel profiler for Chrome/Perfetto
// trace files produced by tinyfuse's benchmarks.
package main

import (
	"encoding/json"
	"fmt"
	"io"
	"math"
	"sort"
	"strings"
)

// Event is one complete ("ph":"X") trace event.
type Event struct {
	Name string                 `json:"name"`
	Cat  string                 `json:"cat"`
	Ph   string                 `json:"ph"`
	TS   float64                `json:"ts"`  // microseconds
	Dur  float64                `json:"dur"` // microseconds
	Args map[string]interface{} `json:"args"`
}

// Trace is a parsed trace file.
type Trace struct {
	Metadata map[string]interface{} `json:"metadata"`
	Events   []Event                `json:"traceEvents"`
}

// ParseTrace accepts either the object form {"traceEvents": [...]} or a bare
// JSON array of events, and keeps only complete events.
func ParseTrace(r io.Reader) (*Trace, error) {
	raw, err := io.ReadAll(r)
	if err != nil {
		return nil, err
	}
	t := &Trace{}
	trimmed := strings.TrimSpace(string(raw))
	if strings.HasPrefix(trimmed, "[") {
		err = json.Unmarshal(raw, &t.Events)
	} else {
		err = json.Unmarshal(raw, t)
	}
	if err != nil {
		return nil, fmt.Errorf("parse trace: %w", err)
	}
	kept := t.Events[:0]
	for _, e := range t.Events {
		if e.Ph == "X" && e.Dur >= 0 {
			kept = append(kept, e)
		}
	}
	t.Events = kept
	return t, nil
}

// MetaFloat returns a numeric metadata field, or 0 if absent.
func (t *Trace) MetaFloat(key string) float64 {
	if v, ok := t.Metadata[key].(float64); ok {
		return v
	}
	return 0
}

func argFloat(e Event, key string) float64 {
	if v, ok := e.Args[key].(float64); ok {
		return v
	}
	return 0
}

// Stat summarizes every launch of one kernel within one variant.
type Stat struct {
	Variant string
	Kernel  string
	Layer   float64
	Count   int
	Total   float64 // us
	Min     float64
	Median  float64
	P95     float64
	FLOPs   float64 // per launch
	Bytes   float64 // per launch (minimum traffic estimate)
}

// GFLOPS is achieved throughput at the median launch time.
func (s Stat) GFLOPS() float64 {
	if s.Median <= 0 {
		return 0
	}
	return s.FLOPs / (s.Median * 1e3)
}

// GBps is achieved memory bandwidth at the median launch time.
func (s Stat) GBps() float64 {
	if s.Median <= 0 {
		return 0
	}
	return s.Bytes / (s.Median * 1e3)
}

// Intensity is arithmetic intensity in FLOPs per byte.
func (s Stat) Intensity() float64 {
	if s.Bytes <= 0 {
		return 0
	}
	return s.FLOPs / s.Bytes
}

// Percentile uses linear interpolation between closest ranks; xs must be sorted.
func Percentile(xs []float64, p float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	if len(xs) == 1 {
		return xs[0]
	}
	pos := p / 100 * float64(len(xs)-1)
	lo := int(math.Floor(pos))
	hi := int(math.Ceil(pos))
	frac := pos - float64(lo)
	return xs[lo]*(1-frac) + xs[hi]*frac
}

// Aggregate groups events by (variant, kernel name).
func Aggregate(events []Event) []Stat {
	type key struct{ variant, kernel string }
	durs := map[key][]float64{}
	first := map[key]Event{}
	var order []key
	for _, e := range events {
		variant := e.Cat
		if v, ok := e.Args["variant"].(string); ok && v != "" {
			variant = v
		}
		k := key{variant, e.Name}
		if _, seen := durs[k]; !seen {
			order = append(order, k)
			first[k] = e
		}
		durs[k] = append(durs[k], e.Dur)
	}
	stats := make([]Stat, 0, len(order))
	for _, k := range order {
		d := durs[k]
		sort.Float64s(d)
		total := 0.0
		for _, x := range d {
			total += x
		}
		e := first[k]
		layer := argFloat(e, "layer")
		if layer == 0 {
			layer = argFloat(e, "node")
		}
		stats = append(stats, Stat{
			Variant: k.variant, Kernel: k.kernel, Layer: layer, Count: len(d),
			Total: total, Min: d[0], Median: Percentile(d, 50), P95: Percentile(d, 95),
			FLOPs: argFloat(e, "flops"), Bytes: argFloat(e, "bytes"),
		})
	}
	return stats
}

// Roofline describes the machine ceilings used to judge a kernel.
type Roofline struct {
	PeakGFLOPS float64 // compute ceiling
	BWGBps     float64 // memory bandwidth ceiling
}

// Ridge is the arithmetic intensity where the two ceilings meet.
func (r Roofline) Ridge() float64 {
	if r.BWGBps <= 0 {
		return 0
	}
	return r.PeakGFLOPS / r.BWGBps
}

// Attainable is the roofline bound for a kernel with the given intensity.
func (r Roofline) Attainable(intensity float64) float64 {
	if r.BWGBps <= 0 {
		return r.PeakGFLOPS
	}
	return math.Min(r.PeakGFLOPS, intensity*r.BWGBps)
}

// Bound classifies a kernel as compute- or memory-bound.
func (r Roofline) Bound(intensity float64) string {
	if r.BWGBps <= 0 || r.PeakGFLOPS <= 0 {
		return "-"
	}
	if intensity >= r.Ridge() {
		return "compute"
	}
	return "memory"
}

// VariantTotal is the summed median time per variant (one forward pass when
// each kernel runs once per pass).
func VariantTotals(stats []Stat) map[string]float64 {
	out := map[string]float64{}
	for _, s := range stats {
		out[s.Variant] += s.Median
	}
	return out
}

// Speedup compares two variants layer by layer (matching on Layer).
type Speedup struct {
	Layer          float64
	Kernel         string
	BaseUS, TestUS float64
}

// Compare matches kernels of variant a and b by layer index.
func Compare(stats []Stat, a, b string) []Speedup {
	base := map[float64]Stat{}
	for _, s := range stats {
		if s.Variant == a {
			base[s.Layer] = s
		}
	}
	var out []Speedup
	for _, s := range stats {
		if s.Variant != b {
			continue
		}
		if bs, ok := base[s.Layer]; ok {
			out = append(out, Speedup{Layer: s.Layer, Kernel: s.Kernel, BaseUS: bs.Median, TestUS: s.Median})
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Layer < out[j].Layer })
	return out
}
