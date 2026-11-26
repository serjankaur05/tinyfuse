package main

import (
	"bytes"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const sample = `{"metadata":{"isa":"avx512","peak_gflops_1t":200,"peak_gflops_mt":400},
"traceEvents":[
{"name":"L1 linear","cat":"naive","ph":"X","ts":0,"dur":100,"args":{"layer":1,"variant":"naive","flops":1000000,"bytes":100000}},
{"name":"L1 linear","cat":"naive","ph":"X","ts":100,"dur":300,"args":{"layer":1,"variant":"naive","flops":1000000,"bytes":100000}},
{"name":"L1 linear","cat":"naive","ph":"X","ts":400,"dur":200,"args":{"layer":1,"variant":"naive","flops":1000000,"bytes":100000}},
{"name":"L1 linear","cat":"fast_omp","ph":"X","ts":600,"dur":10,"args":{"layer":1,"variant":"fast_omp","flops":1000000,"bytes":100000}},
{"name":"meta","ph":"M","ts":0}
]}`

func TestParseKeepsOnlyCompleteEvents(t *testing.T) {
	tr, err := ParseTrace(strings.NewReader(sample))
	if err != nil {
		t.Fatal(err)
	}
	if len(tr.Events) != 4 {
		t.Fatalf("want 4 complete events, got %d", len(tr.Events))
	}
	if tr.MetaFloat("peak_gflops_1t") != 200 {
		t.Fatal("metadata not parsed")
	}
}

func TestParseBareArray(t *testing.T) {
	tr, err := ParseTrace(strings.NewReader(`[{"name":"k","ph":"X","dur":5}]`))
	if err != nil || len(tr.Events) != 1 {
		t.Fatalf("bare array: %v %d", err, len(tr.Events))
	}
}

func TestPercentile(t *testing.T) {
	xs := []float64{1, 2, 3, 4}
	cases := map[float64]float64{0: 1, 50: 2.5, 100: 4, 95: 3.85}
	for p, want := range cases {
		if got := Percentile(xs, p); math.Abs(got-want) > 1e-9 {
			t.Errorf("p%v = %v, want %v", p, got, want)
		}
	}
	if Percentile(nil, 50) != 0 {
		t.Error("empty slice should give 0")
	}
}

func TestAggregateAndThroughput(t *testing.T) {
	tr, _ := ParseTrace(strings.NewReader(sample))
	stats := Aggregate(tr.Events)
	if len(stats) != 2 {
		t.Fatalf("want 2 groups, got %d", len(stats))
	}
	naive := stats[0]
	if naive.Count != 3 || naive.Median != 200 || naive.Min != 100 || naive.Total != 600 {
		t.Fatalf("bad naive stat: %+v", naive)
	}
	// 1e6 FLOPs in 200 us = 5 GFLOP/s; 1e5 bytes in 200 us = 0.5 GB/s
	if math.Abs(naive.GFLOPS()-5) > 1e-9 || math.Abs(naive.GBps()-0.5) > 1e-9 {
		t.Fatalf("throughput: %v GFLOP/s %v GB/s", naive.GFLOPS(), naive.GBps())
	}
	if naive.Intensity() != 10 {
		t.Fatalf("intensity %v", naive.Intensity())
	}
}

func TestRoofline(t *testing.T) {
	r := Roofline{PeakGFLOPS: 400, BWGBps: 20}
	if r.Ridge() != 20 {
		t.Fatalf("ridge %v", r.Ridge())
	}
	if r.Bound(5) != "memory" || r.Bound(50) != "compute" {
		t.Fatal("bound classification")
	}
	if r.Attainable(5) != 100 || r.Attainable(50) != 400 {
		t.Fatal("attainable")
	}
}

func TestCompareAndPeakSelection(t *testing.T) {
	tr, _ := ParseTrace(strings.NewReader(sample))
	stats := Aggregate(tr.Events)
	c := Compare(stats, "naive", "fast_omp")
	if len(c) != 1 || c[0].BaseUS/c[0].TestUS != 20 {
		t.Fatalf("compare: %+v", c)
	}
	if peakFor(tr, "fast_omp", 0) != 400 || peakFor(tr, "naive", 0) != 200 || peakFor(tr, "x", 7) != 7 {
		t.Fatal("peak selection")
	}
}

func TestRunWritesReport(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "t.json")
	os.WriteFile(path, []byte(sample), 0o644)
	html := filepath.Join(dir, "r.html")
	var out bytes.Buffer
	err := run(options{tracePath: path, compare: "naive,fast_omp", top: 2, html: html}, &out)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out.String(), "20.0x") {
		t.Fatalf("missing speedup in output:\n%s", out.String())
	}
	b, _ := os.ReadFile(html)
	if !strings.Contains(string(b), "<svg") || !strings.Contains(string(b), "fast_omp") {
		t.Fatal("html report incomplete")
	}
}
