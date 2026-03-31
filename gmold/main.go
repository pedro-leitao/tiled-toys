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

type paletteType int
type renderEngine int

const (
	palettePlasma paletteType = iota
	paletteForest
	paletteAurora
	paletteMono
	paletteViridis
)

const (
	renderEngineAuto renderEngine = iota
	renderEngineCPU
	renderEngineGPU
)

type paletteStop struct {
	t float64
	c color.RGBA
}

type agent struct {
	x     float64
	y     float64
	angle float64
}

type moldSim struct {
	width, height int
	trail         []float32
	scratch       []float32
	agents        []agent
	rnd           *rand.Rand

	sensorDist  float64
	sensorAngle float64
	turnAngle   float64
	moveSpeed   float64
	deposit     float64
	diffusion   float64
	decay       float64
	jitter      float64
	persistence float64
	gpu         *gpuMold

	step int
}

type gpuMold struct {
	instance        *wgpu.Instance
	adapter         *wgpu.Adapter
	device          *wgpu.Device
	queue           *wgpu.Queue
	pipeline        *wgpu.ComputePipeline
	bindGroupLayout *wgpu.BindGroupLayout
	paramsBuffer    *wgpu.Buffer
	trailBuffer     *wgpu.Buffer
	trailNextBuffer *wgpu.Buffer
	readbackBuffer  *wgpu.Buffer
	bindGroup       *wgpu.BindGroup
	bindGroupAlt    *wgpu.BindGroup
	bufferSize      uint64
	width           int
	height          int
	useAlt          bool
}

type gpuMoldParams struct {
	Width     uint32
	Height    uint32
	Diffusion float32
	Decay     float32
}

var framePNGEncoder = png.Encoder{CompressionLevel: png.BestSpeed}
var frameBuffer bytes.Buffer
var frameBase64Buffer []byte

func main() {
	spm := flag.Int("spm", 900, "Simulation steps per minute")
	cellSize := flag.Int("cell-size", 4, "Cell size in pixels")
	agentCount := flag.Int("agents", 12000, "Number of slime mold agents")
	substeps := flag.Int("substeps", 2, "Simulation updates per tick")
	sensorDist := flag.Float64("sensor-distance", 8.0, "Sensor distance in grid cells")
	sensorAngleDeg := flag.Float64("sensor-angle", 32.0, "Sensor angle in degrees")
	turnAngleDeg := flag.Float64("turn-angle", 26.0, "Turn angle in degrees")
	moveSpeed := flag.Float64("move-speed", 0.72, "Agent movement speed in cells per step")
	deposit := flag.Float64("deposit", 0.045, "Trail deposit amount per step")
	diffusion := flag.Float64("diffusion", 0.16, "Trail diffusion amount per step")
	decay := flag.Float64("decay", 0.010, "Trail decay amount per step")
	jitterDeg := flag.Float64("jitter", 2.2, "Per-step directional jitter in degrees")
	persistence := flag.Float64("persistence", 0.86, "Directional persistence 0..0.99 (higher = smoother, vein-like flow)")
	paletteName := flag.String("palette", "plasma", "Color palette: plasma|forest|aurora|mono|viridis")
	engineName := flag.String("engine", "auto", "Computation engine: auto|cpu|gpu")
	frameStride := flag.Int("frame-stride", 1, "Render one frame every N simulation steps")
	flag.Parse()

	if *spm < 1 {
		*spm = 1
	}
	if *cellSize < 2 {
		*cellSize = 2
	}
	if *cellSize > 14 {
		*cellSize = 14
	}
	if *agentCount < 100 {
		*agentCount = 100
	}
	if *agentCount > 200000 {
		*agentCount = 200000
	}
	if *substeps < 1 {
		*substeps = 1
	}
	if *substeps > 12 {
		*substeps = 12
	}
	if *sensorDist < 1 {
		*sensorDist = 1
	}
	if *sensorDist > 32 {
		*sensorDist = 32
	}
	if *moveSpeed <= 0 {
		*moveSpeed = 0.72
	}
	if *deposit <= 0 {
		*deposit = 0.045
	}
	if *diffusion < 0 {
		*diffusion = 0
	}
	if *diffusion > 1 {
		*diffusion = 1
	}
	if *decay < 0 {
		*decay = 0
	}
	if *decay > 0.35 {
		*decay = 0.35
	}
	if *frameStride < 1 {
		*frameStride = 1
	}
	if *jitterDeg < 0 {
		*jitterDeg = 0
	}
	if *persistence < 0 {
		*persistence = 0
	}
	if *persistence > 0.99 {
		*persistence = 0.99
	}

	sim := &moldSim{
		rnd:         rand.New(rand.NewSource(time.Now().UnixNano())),
		sensorDist:  *sensorDist,
		sensorAngle: degToRad(*sensorAngleDeg),
		turnAngle:   degToRad(*turnAngleDeg),
		moveSpeed:   *moveSpeed,
		deposit:     *deposit,
		diffusion:   *diffusion,
		decay:       *decay,
		jitter:      degToRad(*jitterDeg),
		persistence: *persistence,
	}
	palette := parsePalette(*paletteName)
	engine := parseRenderEngine(*engineName)
	if engine != renderEngineCPU {
		gpu, err := newGPUMold()
		if err != nil {
			if engine == renderEngineGPU {
				fmt.Fprintf(os.Stderr, "failed to initialize GPU engine: %v\n", err)
				os.Exit(1)
			}
			fmt.Fprintf(os.Stderr, "GPU engine unavailable (%v), falling back to CPU\n", err)
		} else {
			sim.gpu = gpu
			fmt.Fprintln(os.Stderr, "using WebGPU mold field engine")
			defer sim.gpu.Close()
		}
	}

	wpx, hpx := getTermPixels()
	gridW := max(32, wpx/(*cellSize))
	gridH := max(32, hpx/(*cellSize))
	sim.Resize(gridW, gridH, *agentCount)
	sim.ResetField()
	sim.ScatterAgents()

	c := make(chan os.Signal, 1)
	signal.Notify(c, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-c
		cleanupTerminal()
		os.Exit(0)
	}()

	fmt.Print("\033[?25l\033[2J\033[H")
	fmt.Print("\033_Ga=d,d=A,q=2\033\\")
	defer cleanupTerminal()

	ticker := time.NewTicker(time.Minute / time.Duration(*spm))
	defer ticker.Stop()
	resizeTicker := time.NewTicker(700 * time.Millisecond)
	defer resizeTicker.Stop()

	step := 0
	currentID := 1
	previousID := 2

	render := func() {
		wpx, hpx = getTermPixels()
		newGridW := max(32, wpx/(*cellSize))
		newGridH := max(32, hpx/(*cellSize))
		if newGridW != sim.width || newGridH != sim.height {
			sim.Resize(newGridW, newGridH, *agentCount)
			sim.ResetField()
			sim.ScatterAgents()
		}

		img := image.NewRGBA(image.Rect(0, 0, wpx, hpx))
		drawTrail(img, sim, palette, *cellSize)

		fmt.Print("\033[H")
		printKittyImage(img, currentID)
		fmt.Printf("\033_Ga=d,d=i,q=2,i=%d\033\\", previousID)
		currentID, previousID = previousID, currentID
	}

	render()

	for {
		select {
		case <-ticker.C:
			for i := 0; i < *substeps; i++ {
				sim.Step()
			}
			step++
			if step%720 == 0 && sim.meanTrail() < 0.0018 {
				sim.ResetField()
				sim.ScatterAgents()
			}
			if step%*frameStride == 0 {
				render()
			}
		case <-resizeTicker.C:
			render()
		}
	}
}

func (s *moldSim) Resize(w, h, agentCount int) {
	s.width = w
	s.height = h
	n := w * h
	s.trail = make([]float32, n)
	s.scratch = make([]float32, n)
	s.agents = make([]agent, agentCount)
}

func (s *moldSim) ResetField() {
	for i := range s.trail {
		s.trail[i] = 0
		s.scratch[i] = 0
	}
	// Very sparse low-intensity noise to break perfect symmetry without creating blooms.
	for i := 0; i < (s.width*s.height)/350; i++ {
		x := s.rnd.Intn(s.width)
		y := s.rnd.Intn(s.height)
		s.trail[y*s.width+x] = 0.02 + float32(s.rnd.Float64()*0.03)
	}
	s.step = 0
}

func (s *moldSim) ScatterAgents() {
	for i := range s.agents {
		x := s.rnd.Float64() * float64(s.width)
		y := s.rnd.Float64() * float64(s.height)
		s.agents[i].x = wrapFloat(x, float64(s.width))
		s.agents[i].y = wrapFloat(y, float64(s.height))
		s.agents[i].angle = s.rnd.Float64() * 2 * math.Pi
	}
}

func (s *moldSim) Step() {
	w := float64(s.width)
	h := float64(s.height)

	for i := range s.agents {
		a := &s.agents[i]
		desired := a.angle
		left := s.sample(a.x, a.y, a.angle-s.sensorAngle)
		forward := s.sample(a.x, a.y, a.angle)
		right := s.sample(a.x, a.y, a.angle+s.sensorAngle)

		if forward < left && forward < right {
			if s.rnd.Float64() < 0.5 {
				desired += s.turnAngle
			} else {
				desired -= s.turnAngle
			}
		} else if right > left {
			desired += s.turnAngle
		} else if left > right {
			desired -= s.turnAngle
		}

		if s.jitter > 0 {
			desired += (s.rnd.Float64()*2 - 1) * s.jitter
		}

		// Heading inertia: preserve some of the previous direction for smoother veins.
		a.angle += angleDelta(a.angle, desired) * (1.0 - s.persistence)

		a.x = wrapFloat(a.x+math.Cos(a.angle)*s.moveSpeed, w)
		a.y = wrapFloat(a.y+math.Sin(a.angle)*s.moveSpeed, h)

		x := int(a.x)
		y := int(a.y)
		idx := y*s.width + x
		v := s.trail[idx] + float32(s.deposit)
		if v > 1 {
			v = 1
		}
		s.trail[idx] = v
	}

	if s.gpu != nil {
		if err := s.gpu.StepAndReadback(s); err != nil {
			fmt.Fprintf(os.Stderr, "GPU step failed: %v\n", err)
			s.gpu.Close()
			s.gpu = nil
			s.diffuseAndDecay()
		}
	} else {
		s.diffuseAndDecay()
	}
	s.step++
}

func (s *moldSim) sample(x, y, angle float64) float64 {
	sx := wrapFloat(x+math.Cos(angle)*s.sensorDist, float64(s.width))
	sy := wrapFloat(y+math.Sin(angle)*s.sensorDist, float64(s.height))
	ix := int(sx)
	iy := int(sy)
	return float64(s.trail[iy*s.width+ix])
}

func (s *moldSim) diffuseAndDecay() {
	for y := 0; y < s.height; y++ {
		ym1 := y - 1
		if ym1 < 0 {
			ym1 = s.height - 1
		}
		yp1 := y + 1
		if yp1 >= s.height {
			yp1 = 0
		}
		row := y * s.width
		rowUp := ym1 * s.width
		rowDn := yp1 * s.width

		for x := 0; x < s.width; x++ {
			xm1 := x - 1
			if xm1 < 0 {
				xm1 = s.width - 1
			}
			xp1 := x + 1
			if xp1 >= s.width {
				xp1 = 0
			}

			center := s.trail[row+x]
			near := s.trail[row+xm1] + s.trail[row+xp1] + s.trail[rowUp+x] + s.trail[rowDn+x]
			mixed := center*(1-float32(s.diffusion)) + (near*0.25)*float32(s.diffusion)
			mixed *= float32(1 - s.decay)
			if mixed < 0 {
				mixed = 0
			}
			s.scratch[row+x] = mixed
		}
	}
	s.trail, s.scratch = s.scratch, s.trail
}

func (s *moldSim) meanTrail() float64 {
	if len(s.trail) == 0 {
		return 0
	}
	var sum float64
	for _, v := range s.trail {
		sum += float64(v)
	}
	return sum / float64(len(s.trail))
}

func (s *moldSim) dropBlob(cx, cy, radius int, amount float32) {
	r2 := radius * radius
	for dy := -radius; dy <= radius; dy++ {
		y := cy + dy
		y = ((y % s.height) + s.height) % s.height
		for dx := -radius; dx <= radius; dx++ {
			if dx*dx+dy*dy > r2 {
				continue
			}
			x := cx + dx
			x = ((x % s.width) + s.width) % s.width
			idx := y*s.width + x
			v := s.trail[idx] + amount
			if v > 1 {
				v = 1
			}
			s.trail[idx] = v
		}
	}
}

func parsePalette(name string) paletteType {
	s := strings.TrimSpace(strings.ToLower(name))
	switch s {
	case "forest":
		return paletteForest
	case "aurora":
		return paletteAurora
	case "mono", "monochrome":
		return paletteMono
	case "viridis":
		return paletteViridis
	default:
		return palettePlasma
	}
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

func paletteStops(p paletteType) []paletteStop {
	switch p {
	case paletteForest:
		return []paletteStop{
			{t: 0.0, c: color.RGBA{R: 3, G: 9, B: 6, A: 255}},
			{t: 0.28, c: color.RGBA{R: 18, G: 58, B: 36, A: 255}},
			{t: 0.60, c: color.RGBA{R: 72, G: 148, B: 78, A: 255}},
			{t: 1.0, c: color.RGBA{R: 200, G: 240, B: 135, A: 255}},
		}
	case paletteAurora:
		return []paletteStop{
			{t: 0.0, c: color.RGBA{R: 5, G: 10, B: 26, A: 255}},
			{t: 0.22, c: color.RGBA{R: 27, G: 26, B: 92, A: 255}},
			{t: 0.55, c: color.RGBA{R: 48, G: 156, B: 173, A: 255}},
			{t: 0.85, c: color.RGBA{R: 125, G: 238, B: 155, A: 255}},
			{t: 1.0, c: color.RGBA{R: 244, G: 255, B: 210, A: 255}},
		}
	case paletteMono:
		return []paletteStop{
			{t: 0.0, c: color.RGBA{R: 6, G: 6, B: 6, A: 255}},
			{t: 0.35, c: color.RGBA{R: 42, G: 42, B: 42, A: 255}},
			{t: 0.7, c: color.RGBA{R: 134, G: 134, B: 134, A: 255}},
			{t: 1.0, c: color.RGBA{R: 252, G: 252, B: 252, A: 255}},
		}
	case paletteViridis:
		return []paletteStop{
			{t: 0.0, c: color.RGBA{R: 13, G: 16, B: 48, A: 255}},
			{t: 0.25, c: color.RGBA{R: 43, G: 71, B: 123, A: 255}},
			{t: 0.5, c: color.RGBA{R: 33, G: 144, B: 140, A: 255}},
			{t: 0.75, c: color.RGBA{R: 94, G: 201, B: 98, A: 255}},
			{t: 1.0, c: color.RGBA{R: 253, G: 231, B: 37, A: 255}},
		}
	default:
		return []paletteStop{
			{t: 0.0, c: color.RGBA{R: 7, G: 5, B: 22, A: 255}},
			{t: 0.2, c: color.RGBA{R: 32, G: 16, B: 88, A: 255}},
			{t: 0.45, c: color.RGBA{R: 121, G: 36, B: 154, A: 255}},
			{t: 0.72, c: color.RGBA{R: 233, G: 94, B: 74, A: 255}},
			{t: 1.0, c: color.RGBA{R: 255, G: 236, B: 132, A: 255}},
		}
	}
}

func drawTrail(img *image.RGBA, sim *moldSim, palette paletteType, cellSize int) {
	stops := paletteStops(palette)
	bg := stops[0].c
	fillBackground(img, bg)

	viewW := min(sim.width, img.Bounds().Dx()/cellSize)
	viewH := min(sim.height, img.Bounds().Dy()/cellSize)
	if viewW <= 0 || viewH <= 0 {
		return
	}

	for y := 0; y < viewH; y++ {
		py := y * cellSize
		for x := 0; x < viewW; x++ {
			idx := y*sim.width + x
			v := float64(sim.trail[idx])
			if v < 0.001 {
				continue
			}
			// Contrast boost to reveal low-intensity trails.
			v = math.Sqrt(v)
			if v > 1 {
				v = 1
			}
			c := sampleGradient(stops, v)
			fillRect(img, x*cellSize, py, cellSize, cellSize, c)
		}
	}
}

func sampleGradient(stops []paletteStop, t float64) color.RGBA {
	if len(stops) == 0 {
		return color.RGBA{A: 255}
	}
	if t <= stops[0].t {
		return stops[0].c
	}
	last := stops[len(stops)-1]
	if t >= last.t {
		return last.c
	}
	for i := 0; i < len(stops)-1; i++ {
		a := stops[i]
		b := stops[i+1]
		if t < a.t || t > b.t {
			continue
		}
		u := (t - a.t) / (b.t - a.t)
		return lerpColor(a.c, b.c, u)
	}
	return last.c
}

func lerpColor(a, b color.RGBA, t float64) color.RGBA {
	if t < 0 {
		t = 0
	}
	if t > 1 {
		t = 1
	}
	return color.RGBA{
		R: uint8(float64(a.R) + (float64(b.R)-float64(a.R))*t),
		G: uint8(float64(a.G) + (float64(b.G)-float64(a.G))*t),
		B: uint8(float64(a.B) + (float64(b.B)-float64(a.B))*t),
		A: 255,
	}
}

func fillBackground(img *image.RGBA, c color.RGBA) {
	b := img.Bounds()
	for y := b.Min.Y; y < b.Max.Y; y++ {
		o := img.PixOffset(b.Min.X, y)
		for x := b.Min.X; x < b.Max.X; x++ {
			img.Pix[o+0] = c.R
			img.Pix[o+1] = c.G
			img.Pix[o+2] = c.B
			img.Pix[o+3] = c.A
			o += 4
		}
	}
}

func fillRect(img *image.RGBA, x, y, w, h int, c color.RGBA) {
	if w <= 0 || h <= 0 {
		return
	}
	bounds := img.Bounds()
	x0 := max(x, bounds.Min.X)
	y0 := max(y, bounds.Min.Y)
	x1 := min(x+w, bounds.Max.X)
	y1 := min(y+h, bounds.Max.Y)
	if x0 >= x1 || y0 >= y1 {
		return
	}
	for yy := y0; yy < y1; yy++ {
		o := img.PixOffset(x0, yy)
		for xx := x0; xx < x1; xx++ {
			img.Pix[o+0] = c.R
			img.Pix[o+1] = c.G
			img.Pix[o+2] = c.B
			img.Pix[o+3] = c.A
			o += 4
		}
	}
}

func newGPUMold() (*gpuMold, error) {
	r := &gpuMold{}
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
		Label: "gmold-diffuse.wgsl",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{
			Code: moldDiffuseWGSL,
		},
	})
	if err != nil {
		r.Close()
		return nil, err
	}
	defer module.Release()

	pipeline, err := r.device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label: "gmold-diffuse-pipeline",
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
		Label: "gmold-params",
		Size:  16,
		Usage: wgpu.BufferUsage_Uniform | wgpu.BufferUsage_CopyDst,
	})
	if err != nil {
		r.Close()
		return nil, err
	}
	r.paramsBuffer = paramsBuffer

	return r, nil
}

func (r *gpuMold) ensureBuffers(width, height int) error {
	if width == r.width && height == r.height && r.bindGroup != nil {
		return nil
	}

	if r.bindGroup != nil {
		r.bindGroup.Release()
		r.bindGroup = nil
	}
	if r.bindGroupAlt != nil {
		r.bindGroupAlt.Release()
		r.bindGroupAlt = nil
	}
	if r.trailBuffer != nil {
		r.trailBuffer.Release()
		r.trailBuffer = nil
	}
	if r.trailNextBuffer != nil {
		r.trailNextBuffer.Release()
		r.trailNextBuffer = nil
	}
	if r.readbackBuffer != nil {
		r.readbackBuffer.Release()
		r.readbackBuffer = nil
	}

	r.bufferSize = uint64(width*height) * 4

	createStorage := func(label string) (*wgpu.Buffer, error) {
		return r.device.CreateBuffer(&wgpu.BufferDescriptor{
			Label: label,
			Size:  r.bufferSize,
			Usage: wgpu.BufferUsage_Storage | wgpu.BufferUsage_CopyDst | wgpu.BufferUsage_CopySrc,
		})
	}

	var err error
	r.trailBuffer, err = createStorage("gmold-trail")
	if err != nil {
		return err
	}
	r.trailNextBuffer, err = createStorage("gmold-trail-next")
	if err != nil {
		return err
	}

	r.readbackBuffer, err = r.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "gmold-readback",
		Size:  r.bufferSize,
		Usage: wgpu.BufferUsage_MapRead | wgpu.BufferUsage_CopyDst,
	})
	if err != nil {
		return err
	}

	r.bindGroup, err = r.device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Layout: r.bindGroupLayout,
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: r.paramsBuffer, Offset: 0, Size: 16},
			{Binding: 1, Buffer: r.trailBuffer, Offset: 0, Size: r.bufferSize},
			{Binding: 2, Buffer: r.trailNextBuffer, Offset: 0, Size: r.bufferSize},
		},
	})
	if err != nil {
		return err
	}

	r.bindGroupAlt, err = r.device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Layout: r.bindGroupLayout,
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: r.paramsBuffer, Offset: 0, Size: 16},
			{Binding: 1, Buffer: r.trailNextBuffer, Offset: 0, Size: r.bufferSize},
			{Binding: 2, Buffer: r.trailBuffer, Offset: 0, Size: r.bufferSize},
		},
	})
	if err != nil {
		return err
	}

	r.width = width
	r.height = height
	r.useAlt = false
	return nil
}

func (r *gpuMold) StepAndReadback(s *moldSim) error {
	if err := r.ensureBuffers(s.width, s.height); err != nil {
		return err
	}
	if len(s.trail) > 0 {
		r.queue.WriteBuffer(r.currentTrailBuffer(), 0, float32Bytes(s.trail))
	}

	params := gpuMoldParams{
		Width:     uint32(s.width),
		Height:    uint32(s.height),
		Diffusion: float32(s.diffusion),
		Decay:     float32(s.decay),
	}
	r.queue.WriteBuffer(r.paramsBuffer, 0, encodeMoldParams(params))

	encoder, err := r.device.CreateCommandEncoder(nil)
	if err != nil {
		return err
	}
	defer encoder.Release()

	pass := encoder.BeginComputePass(&wgpu.ComputePassDescriptor{Label: "gmold-diffuse-pass"})
	pass.SetPipeline(r.pipeline)
	if r.useAlt {
		pass.SetBindGroup(0, r.bindGroupAlt, nil)
	} else {
		pass.SetBindGroup(0, r.bindGroup, nil)
	}

	const workgroup = 8
	groupsX := uint32((s.width + workgroup - 1) / workgroup)
	groupsY := uint32((s.height + workgroup - 1) / workgroup)
	pass.DispatchWorkgroups(groupsX, groupsY, 1)
	if err := pass.End(); err != nil {
		pass.Release()
		return err
	}
	pass.Release()

	encoder.CopyBufferToBuffer(r.nextTrailBuffer(), 0, r.readbackBuffer, 0, r.bufferSize)
	cmd, err := encoder.Finish(nil)
	if err != nil {
		return err
	}
	defer cmd.Release()
	r.queue.Submit(cmd)

	var status wgpu.BufferMapAsyncStatus
	mapped := false
	if err := r.readbackBuffer.MapAsync(wgpu.MapMode_Read, 0, r.bufferSize, func(s wgpu.BufferMapAsyncStatus) {
		status = s
		mapped = true
	}); err != nil {
		return err
	}
	r.device.Poll(true, nil)
	if !mapped {
		return fmt.Errorf("GPU readback callback not received")
	}
	if status != wgpu.BufferMapAsyncStatus_Success {
		return fmt.Errorf("GPU readback map failed: %v", status)
	}

	raw := r.readbackBuffer.GetMappedRange(0, uint(r.bufferSize))
	mappedSlice := wgpu.FromBytes[float32](raw)
	if len(s.trail) != len(mappedSlice) {
		s.trail = make([]float32, len(mappedSlice))
	}
	copy(s.trail, mappedSlice)
	r.readbackBuffer.Unmap()

	r.useAlt = !r.useAlt
	return nil
}

func (r *gpuMold) currentTrailBuffer() *wgpu.Buffer {
	if r.useAlt {
		return r.trailNextBuffer
	}
	return r.trailBuffer
}

func (r *gpuMold) nextTrailBuffer() *wgpu.Buffer {
	if r.useAlt {
		return r.trailBuffer
	}
	return r.trailNextBuffer
}

func (r *gpuMold) Close() {
	if r.bindGroup != nil {
		r.bindGroup.Release()
		r.bindGroup = nil
	}
	if r.bindGroupAlt != nil {
		r.bindGroupAlt.Release()
		r.bindGroupAlt = nil
	}
	if r.readbackBuffer != nil {
		r.readbackBuffer.Release()
		r.readbackBuffer = nil
	}
	if r.trailBuffer != nil {
		r.trailBuffer.Release()
		r.trailBuffer = nil
	}
	if r.trailNextBuffer != nil {
		r.trailNextBuffer.Release()
		r.trailNextBuffer = nil
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

func encodeMoldParams(p gpuMoldParams) []byte {
	buf := bytes.Buffer{}
	_ = binary.Write(&buf, binary.LittleEndian, p)
	return buf.Bytes()
}

func float32Bytes(data []float32) []byte {
	if len(data) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(&data[0])), len(data)*4)
}

const moldDiffuseWGSL = `
struct Params {
	width: u32,
	height: u32,
	diffusion: f32,
	decay: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> trailIn: array<f32>;
@group(0) @binding(2) var<storage, read_write> trailOut: array<f32>;

fn idx(x: u32, y: u32, w: u32) -> u32 {
	return y * w + x;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
	if (gid.x >= params.width || gid.y >= params.height) {
		return;
	}

	let w = params.width;
	let h = params.height;
	let x = gid.x;
	let y = gid.y;

	let xm1 = select(x - 1u, w - 1u, x == 0u);
	let xp1 = select(x + 1u, 0u, x + 1u == w);
	let ym1 = select(y - 1u, h - 1u, y == 0u);
	let yp1 = select(y + 1u, 0u, y + 1u == h);

	let i = idx(x, y, w);
	let center = trailIn[i];
	let near = trailIn[idx(xm1, y, w)] + trailIn[idx(xp1, y, w)] + trailIn[idx(x, ym1, w)] + trailIn[idx(x, yp1, w)];

	var mixed = center * (1.0 - params.diffusion) + (near * 0.25) * params.diffusion;
	mixed = mixed * (1.0 - params.decay);
	trailOut[i] = clamp(mixed, 0.0, 1.0);
}
`

func getTermPixels() (int, int) {
	ws := &Winsize{}
	_, _, err := syscall.Syscall(syscall.SYS_IOCTL, uintptr(syscall.Stdout), uintptr(syscall.TIOCGWINSZ), uintptr(unsafe.Pointer(ws)))
	w := int(ws.Xpixel)
	h := int(ws.Ypixel)
	if err != 0 || w == 0 || h == 0 {
		w = int(ws.Col) * 10
		h = int(ws.Row) * 20
	}
	if w == 0 {
		w = 800
	}
	if h == 0 {
		h = 600
	}
	return w, h - 20
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
	n := base64.StdEncoding.EncodedLen(len(raw))
	if cap(frameBase64Buffer) < n {
		frameBase64Buffer = make([]byte, n)
	}
	enc := frameBase64Buffer[:n]
	base64.StdEncoding.Encode(enc, raw)

	for i := 0; i < len(enc); i += 4096 {
		end := i + 4096
		m := 1
		if end >= len(enc) {
			end = len(enc)
			m = 0
		}
		if i == 0 {
			fmt.Printf("\033_Ga=T,f=100,t=d,q=2,i=%d,m=%d;%s\033\\", id, m, enc[i:end])
		} else {
			fmt.Printf("\033_Gm=%d;%s\033\\", m, enc[i:end])
		}
	}
}

func degToRad(v float64) float64 {
	return v * math.Pi / 180.0
}

func angleDelta(from, to float64) float64 {
	d := math.Mod(to-from+math.Pi, 2*math.Pi)
	if d < 0 {
		d += 2 * math.Pi
	}
	return d - math.Pi
}

func wrapFloat(v, maxV float64) float64 {
	for v < 0 {
		v += maxV
	}
	for v >= maxV {
		v -= maxV
	}
	return v
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
