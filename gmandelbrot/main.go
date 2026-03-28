package main

import (
	"bytes"
	"encoding/base64"
	"encoding/binary"
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"math"
	"math/rand"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"
	"unsafe"

	"github.com/rajveermalviya/go-webgpu/wgpu"
)

type Winsize struct {
	Row    uint16
	Col    uint16
	Xpixel uint16
	Ypixel uint16
}

type FocusPoint struct {
	X, Y float32
	Span float32
}

type Viewer struct {
	centerX float32
	centerY float32
	spanX   float32
	angle   float32

	area       FocusPoint
	refineStep int
	refineMax  int
	holdStep   int
	holdMax    int
	maxBlock   int

	iterBase int
	iterMax  int
	mathMode fractalMathMode
	palette  paletteType
	engine   renderEngine
	gpu      *gpuRenderer

	rnd *rand.Rand
}

type paletteType int
type fractalMathMode int
type renderEngine int

type gpuRenderer struct {
	instance        *wgpu.Instance
	adapter         *wgpu.Adapter
	device          *wgpu.Device
	queue           *wgpu.Queue
	pipeline        *wgpu.ComputePipeline
	bindGroupLayout *wgpu.BindGroupLayout
	paramsBuffer    *wgpu.Buffer
	outputBuffer    *wgpu.Buffer
	readbackBuffer  *wgpu.Buffer
	bindGroup       *wgpu.BindGroup
	bufferSize      uint64
	width           int
	height          int
}

const (
	paletteTwilight paletteType = iota
	paletteFire
	paletteIce
	paletteForest
	paletteMono
)

const (
	mathModeFloat fractalMathMode = iota
	mathModeFixed
)

const (
	renderEngineAuto renderEngine = iota
	renderEngineCPU
	renderEngineGPU
)

const (
	fixedShift   = 28
	fixedOne     = int64(1 << fixedShift)
	fixedEscape2 = int64(16) << fixedShift
)

var framePNGEncoder = png.Encoder{CompressionLevel: png.BestSpeed}
var frameBuffer bytes.Buffer
var frameBase64Buffer []byte

var focusPoints = []FocusPoint{
	// Full Overviews (Centered to fit the whole set nicely)
	{X: -0.65, Y: 0.0, Span: 2.1}, // The standard full view
	{X: -1.0, Y: 0.0, Span: 1.2},  // Period-2 bulb and main antenna

	// Seahorse Valley (The cleft between the main cardioid and period-2 bulb)
	// Highly reliable, incredibly rich in spirals
	{X: -0.745, Y: 0.1, Span: 0.05},          // Valley entrance
	{X: -0.748, Y: 0.1, Span: 0.005},         // Medium zoom into the seahorses
	{X: -0.7488, Y: 0.093, Span: 0.0005},     // Deep spiral (Safe for float32)
	{X: -0.74364, Y: 0.13182, Span: 0.00005}, // Extreme micro-detail boundary

	// Elephant Valley (The right-side cleft of the main cardioid)
	// Features branching, tree-like structures rather than spirals
	{X: 0.275, Y: 0.005, Span: 0.05},  // Valley overview
	{X: 0.274, Y: 0.006, Span: 0.005}, // Mid-level elephant trunks
	{X: 0.282, Y: 0.010, Span: 0.002}, // Deep trunk detail

	// Scepter Valley / Period-3 Bulb Boundary
	{X: -1.25066, Y: 0.02012, Span: 0.01}, // Nice chaotic boundary
	{X: -1.310, Y: 0.030, Span: 0.02},     // Period-3 bulb edge
	{X: -1.360, Y: 0.005, Span: 0.05},     // Scepter valley overview

	// Triple-Spiral Valley / Top edge
	{X: -0.10, Y: 0.90, Span: 0.2},          // Approach to the top boundary
	{X: -0.1011, Y: 0.9563, Span: 0.02},     // Top spiraling chaotic zone
	{X: -0.156520, Y: 1.03225, Span: 0.005}, // High-contrast Misiurewicz point

	// Needle / Filament Detail (The long antenna on the left)
	{X: -1.770, Y: 0.0, Span: 0.05},          // Main antenna context
	{X: -1.768778, Y: 0.001738, Span: 0.005}, // Island on the antenna
	{X: -1.94, Y: 0.0, Span: 0.1},            // Extreme left tip context
	{X: -1.999985, Y: 0.0, Span: 0.00005},    // Deep left tip mini-brot

	// Guaranteed Minibrots (Perfectly centered self-similar copies)
	{X: -1.7687788, Y: 0.0, Span: 0.01},    // Largest minibrot on the needle
	{X: -0.1528, Y: 1.0397, Span: 0.05},    // Top edge minibrot
	{X: -0.79319, Y: 0.16093, Span: 0.005}, // Seahorse valley minibrot
	{X: 0.3533, Y: 0.0988, Span: 0.005},    // Elephant valley minibrot
}

func main() {
	spm := flag.Int("spm", 360, "Steps per minute")
	paletteName := flag.String("palette", "twilight", "Color palette: twilight|fire|ice|forest|mono")
	mathModeName := flag.String("math-mode", "fixed", "Mandelbrot kernel: fixed|float")
	engineName := flag.String("engine", "auto", "Render engine: auto|cpu|gpu")
	refineSteps := flag.Int("refine-steps", 85, "How many steps to refine detail in the selected area")
	holdSteps := flag.Int("hold-steps", 40, "How many steps to hold the fully-refined view before switching area")
	maxBlock := flag.Int("max-block", 12, "Initial coarse block size in pixels for low-resolution rendering")
	iterBase := flag.Int("iter-base", 40, "Base Mandelbrot iteration budget at coarse refinement")
	iterMax := flag.Int("iter-max", 800, "Maximum Mandelbrot iteration budget at full refinement")
	frameStride := flag.Int("frame-stride", 1, "Render one frame every N steps (higher = lower CPU)")
	flag.Parse()

	if *spm < 1 {
		*spm = 1
	}
	if *frameStride < 1 {
		*frameStride = 1
	}
	if *refineSteps < 4 {
		*refineSteps = 4
	}
	if *holdSteps < 0 {
		*holdSteps = 0
	}
	if *maxBlock < 1 {
		*maxBlock = 1
	}
	if *iterBase < 16 {
		*iterBase = 16
	}
	if *iterMax < *iterBase+16 {
		*iterMax = *iterBase + 16
	}

	viewer := &Viewer{
		centerX:   -0.75,
		centerY:   0,
		spanX:     3.2,
		refineMax: *refineSteps,
		holdMax:   *holdSteps,
		maxBlock:  *maxBlock,
		iterBase:  *iterBase,
		iterMax:   *iterMax,
		mathMode:  parseMathMode(*mathModeName),
		palette:   parsePalette(*paletteName),
		engine:    parseRenderEngine(*engineName),
		rnd:       rand.New(rand.NewSource(time.Now().UnixNano())),
	}

	if viewer.engine != renderEngineCPU {
		gpu, err := newGPURenderer()
		if err != nil {
			if viewer.engine == renderEngineGPU {
				fmt.Fprintf(os.Stderr, "failed to initialize GPU renderer: %v\n", err)
				os.Exit(1)
			}
			fmt.Fprintf(os.Stderr, "GPU renderer unavailable (%v), falling back to CPU\n", err)
		} else {
			viewer.gpu = gpu
			fmt.Fprintln(os.Stderr, "using WebGPU compute renderer")
			defer viewer.gpu.Close()
		}
	}

	viewer.pickNextArea()

	c := make(chan os.Signal, 1)
	signal.Notify(c, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-c
		cleanupTerminal()
		os.Exit(0)
	}()

	fmt.Print("\033[?25l")
	fmt.Print("\033[2J\033[H")
	fmt.Print("\033_Ga=d,d=A,q=2\033\\")
	defer cleanupTerminal()

	tickDuration := time.Minute / time.Duration(*spm)
	ticker := time.NewTicker(tickDuration)
	defer ticker.Stop()

	resizeTicker := time.NewTicker(700 * time.Millisecond)
	defer resizeTicker.Stop()

	step := 0
	currentID := 1
	previousID := 2

	render := func() {
		w, h := getTermPixels()
		if w < 16 || h < 16 {
			return
		}

		img := viewer.renderFrame(w, h)
		fmt.Print("\033[H")
		printKittyImage(img, currentID)
		fmt.Printf("\033_Ga=d,d=i,q=2,i=%d\033\\", previousID)
		currentID, previousID = previousID, currentID
	}

	render()

	for {
		select {
		case <-ticker.C:
			viewer.advance()
			step++
			if step%*frameStride == 0 {
				render()
			}
		case <-resizeTicker.C:
			render()
		}
	}
}

func (v *Viewer) advance() {
	if v.refineStep < v.refineMax {
		v.refineStep++
		return
	}

	if v.holdStep < v.holdMax {
		v.holdStep++
		return
	}

	v.pickNextArea()
}

func (v *Viewer) pickNextArea() {
	v.area = v.randomInterestingArea()
	v.centerX = v.area.X
	v.centerY = v.area.Y
	v.spanX = v.area.Span
	v.angle = (v.rnd.Float32()*2 - 1) * 0.18
	v.refineStep = 0
	v.holdStep = 0
}

func (v *Viewer) randomInterestingArea() FocusPoint {
	const (
		attempts = 80
		maxIter  = 180
	)

	best := FocusPoint{X: -0.75, Y: 0, Span: 2.1}
	bestScore := -1.0

	for i := 0; i < attempts; i++ {
		span := randomSpan(v.rnd)
		x, y := randomBoundaryBiasedPoint(v.rnd, span)
		score := interestingnessScore(x, y, span, maxIter)
		if score > bestScore {
			bestScore = score
			best = FocusPoint{X: x, Y: y, Span: span}
		}
	}

	return best
}

func randomSpan(rnd *rand.Rand) float32 {
	u := rnd.Float64()
	switch {
	case u < 0.10:
		// occasional broad views
		return float32(0.8 + 1.6*rnd.Float64())
	case u < 0.65:
		// medium zooms
		logMin, logMax := math.Log(0.003), math.Log(0.2)
		t := math.Pow(rnd.Float64(), 0.8)
		return float32(math.Exp(logMin + (logMax-logMin)*t))
	default:
		// deep zooms
		logMin, logMax := math.Log(0.00002), math.Log(0.012)
		t := math.Pow(rnd.Float64(), 1.25)
		return float32(math.Exp(logMin + (logMax-logMin)*t))
	}
}

func randomBoundaryBiasedPoint(rnd *rand.Rand, span float32) (float32, float32) {
	var bx, by float32
	p := rnd.Float64()

	if p < 0.55 {
		// Main cardioid boundary parameterization:
		// c(t) = e^{it}/2 - e^{2it}/4
		t := rnd.Float64() * 2 * math.Pi
		bx = float32(0.5*math.Cos(t) - 0.25*math.Cos(2*t))
		by = float32(0.5*math.Sin(t) - 0.25*math.Sin(2*t))
	} else if p < 0.85 {
		// Period-2 bulb boundary centered at (-1,0), radius 1/4
		t := rnd.Float64() * 2 * math.Pi
		bx = float32(-1.0 + 0.25*math.Cos(t))
		by = float32(0.25 * math.Sin(t))
	} else {
		// exploration fallback in common Mandelbrot viewport
		bx = float32(-2.1 + rnd.Float64()*2.9)
		by = float32(-1.25 + rnd.Float64()*2.5)
	}

	// jitter around boundary seed; scales with zoom
	jit := span * float32(0.8+4.0*rnd.Float64())
	a := float32(rnd.Float64() * 2 * math.Pi)
	x := bx + jit*float32(math.Cos(float64(a)))
	y := by + jit*float32(math.Sin(float64(a)))

	if x < -2.2 {
		x = -2.2
	}
	if x > 0.9 {
		x = 0.9
	}
	if y < -1.35 {
		y = -1.35
	}
	if y > 1.35 {
		y = 1.35
	}

	return x, y
}

func interestingnessScore(cx, cy, span float32, maxIter int) float64 {
	step := span * 0.34
	samples := [9][2]float32{
		{-step, -step}, {0, -step}, {step, -step},
		{-step, 0}, {0, 0}, {step, 0},
		{-step, step}, {0, step}, {step, step},
	}

	vals := [9]int{}
	minIter, maxSeen := maxIter, 0
	escaped := 0
	mean := 0.0

	for i := range samples {
		it := quickEscapeIter(cx+samples[i][0], cy+samples[i][1], maxIter)
		vals[i] = it
		mean += float64(it)
		if it < minIter {
			minIter = it
		}
		if it > maxSeen {
			maxSeen = it
		}
		if it < maxIter {
			escaped++
		}
	}

	mean /= float64(len(vals))
	variance := 0.0
	for i := range vals {
		d := float64(vals[i]) - mean
		variance += d * d
	}
	variance /= float64(len(vals))

	escapedRatio := float64(escaped) / float64(len(vals))
	edgeBalance := 1.0 - math.Abs(escapedRatio-0.5)*2.0
	if edgeBalance < 0 {
		edgeBalance = 0
	}

	varNorm := variance / float64(maxIter*maxIter)
	rangeNorm := float64(maxSeen-minIter) / float64(maxIter)

	// Heuristic formula favoring boundary complexity and local variation.
	return 0.65*varNorm + 0.20*edgeBalance + 0.15*rangeNorm
}

func quickEscapeIter(x, y float32, maxIter int) int {
	zr := float32(0)
	zi := float32(0)
	for iter := 0; iter < maxIter; iter++ {
		zr2 := zr * zr
		zi2 := zi * zi
		if zr2+zi2 > 16 {
			return iter
		}
		zi = 2*zr*zi + y
		zr = zr2 - zi2 + x
	}
	return maxIter
}

func (v *Viewer) renderFrame(w, h int) image.Image {
	progress := float32(v.refineStep) / float32(v.refineMax)
	if progress < 0 {
		progress = 0
	}
	if progress > 1 {
		progress = 1
	}

	if v.gpu != nil {
		img, err := v.renderFrameGPU(w, h, progress)
		if err == nil {
			return img
		}
		fmt.Fprintf(os.Stderr, "GPU render failed (%v), falling back to CPU\n", err)
		v.gpu.Close()
		v.gpu = nil
	}

	img := image.NewRGBA(image.Rect(0, 0, w, h))
	block := blockSizeFromProgress(progress, v.maxBlock)
	invW := 1.0 / float32(w)
	invH := 1.0 / float32(h)
	spanY := v.spanX * (float32(h) * invW)
	rotSin := float32(math.Sin(float64(v.angle)))
	rotCos := float32(math.Cos(float64(v.angle)))
	maxIter := v.currentMaxIter(progress)
	sampleColor := v.mandelbrotColorFixed
	if v.mathMode == mathModeFloat {
		sampleColor = v.mandelbrotColorFloat
	}

	for by := 0; by < h; by += block {
		yEnd := by + block
		if yEnd > h {
			yEnd = h
		}

		sy := by + (yEnd-by)/2
		dy := ((float32(sy) * invH) - 0.5) * spanY

		for bx := 0; bx < w; bx += block {
			xEnd := bx + block
			if xEnd > w {
				xEnd = w
			}

			sx := bx + (xEnd-bx)/2
			dx := ((float32(sx) * invW) - 0.5) * v.spanX
			rx := dx*rotCos - dy*rotSin
			ry := dx*rotSin + dy*rotCos
			x := v.centerX + rx
			y := v.centerY + ry
			col := sampleColor(x, y, maxIter)

			for py := by; py < yEnd; py++ {
				for px := bx; px < xEnd; px++ {
					img.Set(px, py, col)
				}
			}
		}
	}

	return img
}

func (v *Viewer) renderFrameGPU(w, h int, progress float32) (image.Image, error) {
	spanY := v.spanX * (float32(h) / float32(w))
	maxIter := v.currentMaxIter(progress)
	return v.gpu.Render(w, h, gpuFrameParams{
		CenterX: v.centerX,
		CenterY: v.centerY,
		SpanX:   v.spanX,
		SpanY:   spanY,
		RotCos:  float32(math.Cos(float64(v.angle))),
		RotSin:  float32(math.Sin(float64(v.angle))),
		MaxIter: uint32(maxIter),
		Palette: uint32(v.palette),
	})
}

func (v *Viewer) currentMaxIter(progress float32) int {
	iter := int(math.Round(float64(v.iterBase) + float64(progress)*float64(v.iterMax-v.iterBase)))
	if iter < 16 {
		iter = 16
	}
	return iter
}

func blockSizeFromProgress(progress float32, maxBlock int) int {
	if maxBlock <= 1 {
		return 1
	}
	block := int(math.Round(1 + float64(1-progress)*float64(maxBlock-1)))
	if block < 1 {
		return 1
	}
	return block
}

func (v *Viewer) mandelbrotColorFloat(x, y float32, maxIter int) color.RGBA {
	zr := float32(0)
	zi := float32(0)
	escape2 := float32(16)
	iter := 0

	for ; iter < maxIter; iter++ {
		zr2 := zr * zr
		zi2 := zi * zi
		if zr2+zi2 > escape2 {
			break
		}
		zi = 2*zr*zi + y
		zr = zr2 - zi2 + x
	}

	if iter == maxIter {
		return color.RGBA{R: 2, G: 2, B: 3, A: 255}
	}

	smooth := smoothedIter(iter, zr, zi)
	return v.colorFromPalette(smooth, maxIter)
}

func (v *Viewer) mandelbrotColorFixed(x, y float32, maxIter int) color.RGBA {
	cx := toFixed(x)
	cy := toFixed(y)
	var zr int64
	var zi int64
	iter := 0

	for ; iter < maxIter; iter++ {
		zr2 := (zr * zr) >> fixedShift
		zi2 := (zi * zi) >> fixedShift
		if zr2+zi2 > fixedEscape2 {
			break
		}
		zi = ((zr * zi) >> (fixedShift - 1)) + cy
		zr = zr2 - zi2 + cx
	}

	if iter == maxIter {
		return color.RGBA{R: 2, G: 2, B: 3, A: 255}
	}

	return v.colorFromPalette(float64(iter), maxIter)
}

func toFixed(v float32) int64 {
	fv := float64(v) * float64(fixedOne)
	if fv >= 0 {
		return int64(fv + 0.5)
	}
	return int64(fv - 0.5)
}

func smoothedIter(iter int, zr, zi float32) float64 {
	mag2 := float64(zr*zr + zi*zi)
	if mag2 <= 1.000002 {
		return float64(iter)
	}
	logAbs := 0.5 * math.Log(mag2)
	if logAbs <= 0 {
		return float64(iter)
	}
	return float64(iter) + 1.0 - math.Log(logAbs)/math.Log(2.0)
}

func parsePalette(s string) paletteType {
	s = strings.TrimSpace(strings.ToLower(s))
	switch s {
	case "fire":
		return paletteFire
	case "ice":
		return paletteIce
	case "forest":
		return paletteForest
	case "mono", "monochrome":
		return paletteMono
	default:
		return paletteTwilight
	}
}

func parseMathMode(s string) fractalMathMode {
	s = strings.TrimSpace(strings.ToLower(s))
	if s == "float" || s == "float32" {
		return mathModeFloat
	}
	return mathModeFixed
}

func parseRenderEngine(s string) renderEngine {
	s = strings.TrimSpace(strings.ToLower(s))
	switch s {
	case "cpu":
		return renderEngineCPU
	case "gpu", "webgpu", "wgpu":
		return renderEngineGPU
	default:
		return renderEngineAuto
	}
}

type gpuFrameParams struct {
	CenterX float32
	CenterY float32
	SpanX   float32
	SpanY   float32
	RotCos  float32
	RotSin  float32
	MaxIter uint32
	Palette uint32
}

func newGPURenderer() (*gpuRenderer, error) {
	r := &gpuRenderer{}
	r.instance = wgpu.CreateInstance(nil)
	if r.instance == nil {
		return nil, fmt.Errorf("wgpu instance creation failed")
	}

	adapter, err := r.instance.RequestAdapter(nil)
	if err != nil {
		r.Close()
		return nil, err
	}
	r.adapter = adapter

	device, err := r.adapter.RequestDevice(nil)
	if err != nil {
		r.Close()
		return nil, err
	}
	r.device = device
	r.queue = r.device.GetQueue()

	module, err := r.device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label: "mandelbrot-compute.wgsl",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{
			Code: mandelbrotComputeWGSL,
		},
	})
	if err != nil {
		r.Close()
		return nil, err
	}
	defer module.Release()

	pipeline, err := r.device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label: "mandelbrot-compute-pipeline",
		Compute: wgpu.ProgrammableStageDescriptor{
			Module:     module,
			EntryPoint: "main",
		},
	})
	if err != nil {
		r.Close()
		return nil, err
	}
	r.pipeline = pipeline
	r.bindGroupLayout = r.pipeline.GetBindGroupLayout(0)

	paramsBuffer, err := r.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "mandelbrot-params",
		Size:  48,
		Usage: wgpu.BufferUsage_Uniform | wgpu.BufferUsage_CopyDst,
	})
	if err != nil {
		r.Close()
		return nil, err
	}
	r.paramsBuffer = paramsBuffer

	return r, nil
}

func (r *gpuRenderer) ensureBuffers(width, height int) error {
	if width == r.width && height == r.height && r.bindGroup != nil {
		return nil
	}

	if r.bindGroup != nil {
		r.bindGroup.Release()
		r.bindGroup = nil
	}
	if r.outputBuffer != nil {
		r.outputBuffer.Release()
		r.outputBuffer = nil
	}
	if r.readbackBuffer != nil {
		r.readbackBuffer.Release()
		r.readbackBuffer = nil
	}

	bufferSize := uint64(width*height) * uint64(4)

	outputBuffer, err := r.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "mandelbrot-output",
		Size:  bufferSize,
		Usage: wgpu.BufferUsage_Storage | wgpu.BufferUsage_CopySrc,
	})
	if err != nil {
		return err
	}
	r.outputBuffer = outputBuffer

	readbackBuffer, err := r.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "mandelbrot-readback",
		Size:  bufferSize,
		Usage: wgpu.BufferUsage_MapRead | wgpu.BufferUsage_CopyDst,
	})
	if err != nil {
		return err
	}
	r.readbackBuffer = readbackBuffer

	bindGroup, err := r.device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Layout: r.bindGroupLayout,
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: r.paramsBuffer, Size: 48},
			{Binding: 1, Buffer: r.outputBuffer, Size: wgpu.WholeSize},
		},
	})
	if err != nil {
		return err
	}
	r.bindGroup = bindGroup
	r.bufferSize = bufferSize
	r.width = width
	r.height = height
	return nil
}

func (r *gpuRenderer) Render(width, height int, p gpuFrameParams) (image.Image, error) {
	if err := r.ensureBuffers(width, height); err != nil {
		return nil, err
	}

	paramsRaw := make([]byte, 48)
	binary.LittleEndian.PutUint32(paramsRaw[0:], uint32(width))
	binary.LittleEndian.PutUint32(paramsRaw[4:], uint32(height))
	binary.LittleEndian.PutUint32(paramsRaw[8:], p.MaxIter)
	binary.LittleEndian.PutUint32(paramsRaw[12:], p.Palette)
	binary.LittleEndian.PutUint32(paramsRaw[16:], math.Float32bits(p.CenterX))
	binary.LittleEndian.PutUint32(paramsRaw[20:], math.Float32bits(p.CenterY))
	binary.LittleEndian.PutUint32(paramsRaw[24:], math.Float32bits(p.SpanX))
	binary.LittleEndian.PutUint32(paramsRaw[28:], math.Float32bits(p.SpanY))
	binary.LittleEndian.PutUint32(paramsRaw[32:], math.Float32bits(p.RotCos))
	binary.LittleEndian.PutUint32(paramsRaw[36:], math.Float32bits(p.RotSin))
	if err := r.queue.WriteBuffer(r.paramsBuffer, 0, paramsRaw); err != nil {
		return nil, err
	}

	encoder, err := r.device.CreateCommandEncoder(&wgpu.CommandEncoderDescriptor{Label: "mandelbrot-compute-encoder"})
	if err != nil {
		return nil, err
	}
	defer encoder.Release()

	pass := encoder.BeginComputePass(&wgpu.ComputePassDescriptor{Label: "mandelbrot-compute-pass"})
	pass.SetPipeline(r.pipeline)
	pass.SetBindGroup(0, r.bindGroup, nil)

	const workgroup = 8
	groupsX := uint32((width + workgroup - 1) / workgroup)
	groupsY := uint32((height + workgroup - 1) / workgroup)
	pass.DispatchWorkgroups(groupsX, groupsY, 1)
	if err := pass.End(); err != nil {
		pass.Release()
		return nil, err
	}
	pass.Release()

	encoder.CopyBufferToBuffer(r.outputBuffer, 0, r.readbackBuffer, 0, r.bufferSize)

	cmd, err := encoder.Finish(nil)
	if err != nil {
		return nil, err
	}
	defer cmd.Release()
	r.queue.Submit(cmd)

	var status wgpu.BufferMapAsyncStatus
	mapped := false
	if err := r.readbackBuffer.MapAsync(wgpu.MapMode_Read, 0, r.bufferSize, func(s wgpu.BufferMapAsyncStatus) {
		status = s
		mapped = true
	}); err != nil {
		return nil, err
	}
	r.device.Poll(true, nil)
	if !mapped {
		return nil, fmt.Errorf("readback map callback not received")
	}
	if status != wgpu.BufferMapAsyncStatus_Success {
		return nil, fmt.Errorf("readback map failed: %v", status)
	}

	raw := r.readbackBuffer.GetMappedRange(0, uint(r.bufferSize))
	pixels := wgpu.FromBytes[uint32](raw)
	img := image.NewRGBA(image.Rect(0, 0, width, height))
	for i := 0; i < width*height; i++ {
		v := pixels[i]
		o := i * 4
		img.Pix[o+0] = uint8(v)
		img.Pix[o+1] = uint8(v >> 8)
		img.Pix[o+2] = uint8(v >> 16)
		img.Pix[o+3] = uint8(v >> 24)
	}
	r.readbackBuffer.Unmap()

	return img, nil
}

func (r *gpuRenderer) Close() {
	if r.bindGroup != nil {
		r.bindGroup.Release()
		r.bindGroup = nil
	}
	if r.readbackBuffer != nil {
		r.readbackBuffer.Release()
		r.readbackBuffer = nil
	}
	if r.outputBuffer != nil {
		r.outputBuffer.Release()
		r.outputBuffer = nil
	}
	if r.paramsBuffer != nil {
		r.paramsBuffer.Release()
		r.paramsBuffer = nil
	}
	if r.bindGroupLayout != nil {
		r.bindGroupLayout.Release()
		r.bindGroupLayout = nil
	}
	if r.pipeline != nil {
		r.pipeline.Release()
		r.pipeline = nil
	}
	if r.queue != nil {
		r.queue.Release()
		r.queue = nil
	}
	if r.device != nil {
		r.device.Release()
		r.device = nil
	}
	if r.adapter != nil {
		r.adapter.Release()
		r.adapter = nil
	}
	if r.instance != nil {
		r.instance.Release()
		r.instance = nil
	}
}

const mandelbrotComputeWGSL = `
struct Params {
    width: u32,
    height: u32,
    max_iter: u32,
    palette: u32,
    center_x: f32,
    center_y: f32,
    span_x: f32,
    span_y: f32,
	rot_cos: f32,
	rot_sin: f32,
	_pad0: f32,
	_pad1: f32,
};

@group(0) @binding(0)
var<uniform> params: Params;

@group(0) @binding(1)
var<storage, read_write> out_pixels: array<u32>;

fn clamp01(v: f32) -> f32 {
    return clamp(v, 0.0, 1.0);
}

fn color_from_palette(t: f32, palette: u32) -> vec3<f32> {
    let pulse = 0.5 + 0.5 * sin(6.28318 * (t * 1.7 + 0.1));
    let spark = 0.5 + 0.5 * sin(6.28318 * (t * 4.4 + 0.8));

    switch palette {
        case 1u {
            return vec3<f32>(
                clamp01(0.10 + 0.95 * t + 0.20 * spark),
                clamp01(0.02 + 0.45 * t + 0.25 * pulse),
                clamp01(0.01 + 0.12 * t),
            );
        }
        case 2u {
            return vec3<f32>(
                clamp01(0.04 + 0.25 * t + 0.10 * spark),
                clamp01(0.14 + 0.55 * t + 0.20 * pulse),
                clamp01(0.22 + 0.95 * t),
            );
        }
        case 3u {
            return vec3<f32>(
                clamp01(0.03 + 0.20 * t + 0.10 * pulse),
                clamp01(0.10 + 0.75 * t + 0.25 * spark),
                clamp01(0.03 + 0.28 * t),
            );
        }
        case 4u {
            let g = clamp01(0.06 + 0.92 * t + 0.06 * pulse);
            return vec3<f32>(g, g, g);
        }
        default {
            return vec3<f32>(
                clamp01(0.04 + 0.90 * t + 0.15 * spark),
                clamp01(0.04 + 0.35 * t + 0.15 * pulse),
                clamp01(0.10 + 0.95 * (1.0 - t) + 0.10 * spark),
            );
        }
    }
}

fn pack_rgba(color: vec3<f32>) -> u32 {
    let r = u32(round(clamp01(color.x) * 255.0));
    let g = u32(round(clamp01(color.y) * 255.0));
    let b = u32(round(clamp01(color.z) * 255.0));
    return r | (g << 8u) | (b << 16u) | (255u << 24u);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    let idx = gid.y * params.width + gid.x;
    let fx = (f32(gid.x) + 0.5) / f32(params.width);
    let fy = (f32(gid.y) + 0.5) / f32(params.height);

	let dx = (fx - 0.5) * params.span_x;
	let dy = (fy - 0.5) * params.span_y;
	let rx = dx * params.rot_cos - dy * params.rot_sin;
	let ry = dx * params.rot_sin + dy * params.rot_cos;
	let cx = params.center_x + rx;
	let cy = params.center_y + ry;

    var zr = 0.0;
    var zi = 0.0;
    var iter = 0u;

    loop {
        if (iter >= params.max_iter) {
            break;
        }

        let zr2 = zr * zr;
        let zi2 = zi * zi;
        if (zr2 + zi2 > 16.0) {
            break;
        }

        let new_zi = 2.0 * zr * zi + cy;
        let new_zr = zr2 - zi2 + cx;
        zi = new_zi;
        zr = new_zr;
        iter = iter + 1u;
    }

    if (iter == params.max_iter) {
        out_pixels[idx] = 0xFF030202u;
        return;
    }

    let mag2 = zr * zr + zi * zi;
	var smooth_iter = f32(iter);
    if (mag2 > 1.000002) {
        let log_abs = 0.5 * log(mag2);
        if (log_abs > 0.0) {
			smooth_iter = smooth_iter + 1.0 - log(log_abs) / log(2.0);
        }
    }

	let t = clamp01(smooth_iter / f32(params.max_iter));
    let rgb = color_from_palette(t, params.palette);
    out_pixels[idx] = pack_rgba(rgb);
}
`

func (v *Viewer) colorFromPalette(iterSmooth float64, maxIter int) color.RGBA {
	t := iterSmooth / float64(maxIter)
	if t < 0 {
		t = 0
	}
	if t > 1 {
		t = 1
	}

	pulse := 0.5 + 0.5*math.Sin(6.28318*(t*1.7+0.1))
	spark := 0.5 + 0.5*math.Sin(6.28318*(t*4.4+0.8))

	switch v.palette {
	case paletteFire:
		return color.RGBA{
			R: clamp01(0.10 + 0.95*t + 0.20*spark),
			G: clamp01(0.02 + 0.45*t + 0.25*pulse),
			B: clamp01(0.01 + 0.12*t),
			A: 255,
		}
	case paletteIce:
		return color.RGBA{
			R: clamp01(0.04 + 0.25*t + 0.10*spark),
			G: clamp01(0.14 + 0.55*t + 0.20*pulse),
			B: clamp01(0.22 + 0.95*t),
			A: 255,
		}
	case paletteForest:
		return color.RGBA{
			R: clamp01(0.03 + 0.20*t + 0.10*pulse),
			G: clamp01(0.10 + 0.75*t + 0.25*spark),
			B: clamp01(0.03 + 0.28*t),
			A: 255,
		}
	case paletteMono:
		g := clamp01(0.06 + 0.92*t + 0.06*pulse)
		return color.RGBA{R: g, G: g, B: g, A: 255}
	default:
		return color.RGBA{
			R: clamp01(0.04 + 0.90*t + 0.15*spark),
			G: clamp01(0.04 + 0.35*t + 0.15*pulse),
			B: clamp01(0.10 + 0.95*(1.0-t) + 0.10*spark),
			A: 255,
		}
	}
}

func clamp01(v float64) uint8 {
	if v < 0 {
		v = 0
	}
	if v > 1 {
		v = 1
	}
	return uint8(v*255 + 0.5)
}

func getTermPixels() (int, int) {
	ws := &Winsize{}
	_, _, err := syscall.Syscall(syscall.SYS_IOCTL, uintptr(syscall.Stdout), uintptr(syscall.TIOCGWINSZ), uintptr(unsafe.Pointer(ws)))

	width := int(ws.Xpixel)
	height := int(ws.Ypixel)

	if err != 0 || width == 0 || height == 0 {
		width = int(ws.Col) * 10
		height = int(ws.Row) * 20
	}

	if width == 0 {
		width = 800
	}
	if height == 0 {
		height = 600
	}

	return width, height - 20
}

func cleanupTerminal() {
	fmt.Print("\033_Ga=d,d=A,q=2\033\\")
	fmt.Print("\033[?25h")
	fmt.Print("\033[2J\033[H")
}

func printKittyImage(img image.Image, id int) {
	frameBuffer.Reset()
	if err := framePNGEncoder.Encode(&frameBuffer, img); err != nil {
		return
	}
	raw := frameBuffer.Bytes()
	encodedLen := base64.StdEncoding.EncodedLen(len(raw))
	if cap(frameBase64Buffer) < encodedLen {
		frameBase64Buffer = make([]byte, encodedLen)
	}
	encoded := frameBase64Buffer[:encodedLen]
	base64.StdEncoding.Encode(encoded, raw)

	chunkSize := 4096
	for i := 0; i < len(encoded); i += chunkSize {
		end := i + chunkSize
		m := 1
		if end >= len(encoded) {
			end = len(encoded)
			m = 0
		}
		chunk := encoded[i:end]
		if i == 0 {
			fmt.Printf("\033_Ga=T,f=100,t=d,q=2,i=%d,m=%d;%s\033\\", id, m, chunk)
		} else {
			fmt.Printf("\033_Gm=%d;%s\033\\", m, chunk)
		}
	}
}
