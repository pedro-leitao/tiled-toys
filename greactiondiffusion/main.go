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

type paletteStop struct {
	t float64
	c color.RGBA
}

const (
	paletteTwilight paletteType = iota
	paletteFire
	paletteIce
	paletteForest
	paletteMono
	paletteViridis
)

const (
	renderEngineAuto renderEngine = iota
	renderEngineCPU
	renderEngineGPU
)

type rdField struct {
	width          int
	height         int
	u              []float64
	v              []float64
	uNext          []float64
	vNext          []float64
	du             float64
	dv             float64
	feed           float64
	kill           float64
	dt             float64
	injectEvery    int
	injectCount    int
	step           int
	rnd            *rand.Rand
	seedMinRadius  int
	seedMaxRadius  int
	seedCount      int
	seedEdgeJitter float64
	resetMinSteps  int
	resetMaxSteps  int
	resetCooldown  int
	resetThreshold float64
	resetFraction  float64
	resetStride    int
	lastResetStep  int
	resetHitCount  int
	resetHitNeeded int
	stepsSinceReset int

	gpu *gpuRD
}

type gpuRD struct {
	instance        *wgpu.Instance
	adapter         *wgpu.Adapter
	device          *wgpu.Device
	queue           *wgpu.Queue
	pipeline        *wgpu.ComputePipeline
	bindGroupLayout *wgpu.BindGroupLayout
	paramsBuffer    *wgpu.Buffer
	uBuffer         *wgpu.Buffer
	vBuffer         *wgpu.Buffer
	uNextBuffer     *wgpu.Buffer
	vNextBuffer     *wgpu.Buffer
	readbackBuffer  *wgpu.Buffer
	bindGroup       *wgpu.BindGroup
	bindGroupAlt    *wgpu.BindGroup
	bufferSize      uint64
	width           int
	height          int
	useAlt          bool
	uUpload         []float32
	vUpload         []float32
}

var framePNGEncoder = png.Encoder{CompressionLevel: png.BestSpeed}
var frameBuffer bytes.Buffer
var frameBase64Buffer []byte

func main() {
	spm := flag.Int("spm", 720, "Simulation steps per minute")
	cellSize := flag.Int("cell-size", 6, "Cell size in pixels")
	substeps := flag.Int("substeps", 8, "Simulation substeps per tick")
	presetName := flag.String("preset", "coral", "Pattern preset: coral|mitosis|maze|spots|zebra|fingerprint|solitons")
	feed := flag.Float64("feed", -1, "Feed rate (overrides preset when >=0)")
	kill := flag.Float64("kill", -1, "Kill rate (overrides preset when >=0)")
	du := flag.Float64("du", -1, "Diffusion rate of U (overrides preset when >=0)")
	dv := flag.Float64("dv", -1, "Diffusion rate of V (overrides preset when >=0)")
	dt := flag.Float64("dt", 1.0, "Integrator step size")
	injectEvery := flag.Int("inject-every", 450, "Inject new chemical seeds every N steps (0 disables)")
	injectCount := flag.Int("inject-count", 3, "Number of random seed drops per injection")
	paletteName := flag.String("palette", "twilight", "Color palette: twilight|fire|ice|forest|mono|viridis")
	engineName := flag.String("engine", "auto", "Simulation engine: auto|cpu|gpu")
	resetMinSteps := flag.Int("reset-min-steps", 10, "Minimum simulation steps before auto-reset can trigger")
	resetMaxSteps := flag.Int("reset-max-steps", 180000, "Force auto-reset after this many simulation steps (0 disables)")
	frameStride := flag.Int("frame-stride", 1, "Render one frame every N simulation steps")
	flag.Parse()

	if *spm < 1 {
		*spm = 1
	}
	if *cellSize < 2 {
		*cellSize = 2
	}
	if *substeps < 1 {
		*substeps = 1
	}
	if *dt <= 0 {
		*dt = 1.0
	}
	if *injectEvery < 0 {
		*injectEvery = 0
	}
	if *injectCount < 0 {
		*injectCount = 0
	}
	if *frameStride < 1 {
		*frameStride = 1
	}
	if *resetMinSteps < 0 {
		*resetMinSteps = 0
	}
	if *resetMaxSteps < 0 {
		*resetMaxSteps = 0
	}

	params := presetParams(*presetName)
	if *feed >= 0 {
		params.feed = *feed
	}
	if *kill >= 0 {
		params.kill = *kill
	}
	if *du >= 0 {
		params.du = *du
	}
	if *dv >= 0 {
		params.dv = *dv
	}

	rnd := rand.New(rand.NewSource(time.Now().UnixNano()))
	field := &rdField{
		dt:          *dt,
		du:          params.du,
		dv:          params.dv,
		feed:        params.feed,
		kill:        params.kill,
		injectEvery: *injectEvery,
		injectCount: *injectCount,
		rnd:         rnd,
		resetMinSteps:  *resetMinSteps,
		resetMaxSteps:  *resetMaxSteps,
		resetCooldown:  300,
		resetThreshold: 0.3,
		resetFraction:  0.82,
		resetStride:    3,
		resetHitNeeded: 3,
	}

	palette := parsePalette(*paletteName)
	engine := parseRenderEngine(*engineName)
	if engine != renderEngineCPU {
		gpu, err := newGPURD()
		if err != nil {
			if engine == renderEngineGPU {
				fmt.Fprintf(os.Stderr, "failed to initialize GPU engine: %v\n", err)
				os.Exit(1)
			}
			fmt.Fprintf(os.Stderr, "GPU engine unavailable (%v), falling back to CPU\n", err)
		} else {
			field.gpu = gpu
			fmt.Fprintln(os.Stderr, "using WebGPU reaction-diffusion engine")
			defer field.gpu.Close()
		}
	}

	wpx, hpx := getTermPixels()
	gridW := max(32, wpx/(*cellSize))
	gridH := max(32, hpx/(*cellSize))
	field.Resize(gridW, gridH)
	field.Reset()

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
		if newGridW != field.width || newGridH != field.height {
			field.Resize(newGridW, newGridH)
			field.Reset()
		}

		img := image.NewRGBA(image.Rect(0, 0, wpx, hpx))
		if field.gpu != nil {
			if err := field.gpu.ReadbackV(field); err != nil {
				fmt.Fprintf(os.Stderr, "GPU readback failed: %v\n", err)
			}
		}
		viewW := min(field.width, wpx/(*cellSize))
		viewH := min(field.height, hpx/(*cellSize))
		if field.shouldReset(viewW, viewH) {
			field.Reset()
		}
		drawField(img, field, palette, *cellSize)

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
				field.Step()
			}
			step++
			if step%*frameStride == 0 {
				render()
			}
		case <-resizeTicker.C:
			render()
		}
	}
}

type preset struct {
	du   float64
	dv   float64
	feed float64
	kill float64
}

func presetParams(name string) preset {
	switch normalizeName(name) {
	case "mitosis":
		return preset{du: 0.16, dv: 0.08, feed: 0.0367, kill: 0.0649}
	case "maze", "zebra":
		return preset{du: 0.19, dv: 0.05, feed: 0.029, kill: 0.057}
	case "spots":
		return preset{du: 0.16, dv: 0.06, feed: 0.035, kill: 0.065}
	case "fingerprint":
		return preset{du: 0.18, dv: 0.05, feed: 0.032, kill: 0.060}
	case "solitons":
		return preset{du: 0.18, dv: 0.06, feed: 0.030, kill: 0.062}
	default:
		return preset{du: 0.18, dv: 0.05, feed: 0.036, kill: 0.062}
	}
}

func normalizeName(s string) string {
	return strings.ToLower(strings.TrimSpace(s))
}

func parsePalette(name string) paletteType {
	switch normalizeName(name) {
	case "fire":
		return paletteFire
	case "ice":
		return paletteIce
	case "forest":
		return paletteForest
	case "mono", "monochrome":
		return paletteMono
	case "viridis":
		return paletteViridis
	default:
		return paletteTwilight
	}
}

func parseRenderEngine(s string) renderEngine {
	s = normalizeName(s)
	switch s {
	case "cpu":
		return renderEngineCPU
	case "gpu", "webgpu", "wgpu":
		return renderEngineGPU
	default:
		return renderEngineAuto
	}
}

type gpuParams struct {
	Width  uint32
	Height uint32
	Du     float32
	Dv     float32
	Feed   float32
	Kill   float32
	Dt     float32
	Pad    float32
}

func newGPURD() (*gpuRD, error) {
	r := &gpuRD{}
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
		Label: "reaction-diffusion.wgsl",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{
			Code: reactionDiffusionWGSL,
		},
	})
	if err != nil {
		r.Close()
		return nil, err
	}
	defer module.Release()

	pipeline, err := r.device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label: "reaction-diffusion-pipeline",
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
		Label: "reaction-diffusion-params",
		Size:  32,
		Usage: wgpu.BufferUsage_Uniform | wgpu.BufferUsage_CopyDst,
	})
	if err != nil {
		r.Close()
		return nil, err
	}
	r.paramsBuffer = paramsBuffer

	return r, nil
}

func (r *gpuRD) ensureBuffers(width, height int) error {
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
	if r.uBuffer != nil {
		r.uBuffer.Release()
		r.uBuffer = nil
	}
	if r.vBuffer != nil {
		r.vBuffer.Release()
		r.vBuffer = nil
	}
	if r.uNextBuffer != nil {
		r.uNextBuffer.Release()
		r.uNextBuffer = nil
	}
	if r.vNextBuffer != nil {
		r.vNextBuffer.Release()
		r.vNextBuffer = nil
	}
	if r.readbackBuffer != nil {
		r.readbackBuffer.Release()
		r.readbackBuffer = nil
	}

	bufferSize := uint64(width*height) * 4

	createStorage := func(label string) (*wgpu.Buffer, error) {
		return r.device.CreateBuffer(&wgpu.BufferDescriptor{
			Label: label,
			Size:  bufferSize,
			Usage: wgpu.BufferUsage_Storage | wgpu.BufferUsage_CopyDst | wgpu.BufferUsage_CopySrc,
		})
	}

	var err error
	r.uBuffer, err = createStorage("reaction-diffusion-u")
	if err != nil {
		return err
	}
	r.vBuffer, err = createStorage("reaction-diffusion-v")
	if err != nil {
		return err
	}
	r.uNextBuffer, err = createStorage("reaction-diffusion-u-next")
	if err != nil {
		return err
	}
	r.vNextBuffer, err = createStorage("reaction-diffusion-v-next")
	if err != nil {
		return err
	}

	r.readbackBuffer, err = r.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "reaction-diffusion-readback",
		Size:  bufferSize,
		Usage: wgpu.BufferUsage_MapRead | wgpu.BufferUsage_CopyDst,
	})
	if err != nil {
		return err
	}

	r.bindGroup, err = r.device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Layout: r.bindGroupLayout,
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: r.paramsBuffer, Offset: 0, Size: 32},
			{Binding: 1, Buffer: r.uBuffer, Offset: 0, Size: bufferSize},
			{Binding: 2, Buffer: r.vBuffer, Offset: 0, Size: bufferSize},
			{Binding: 3, Buffer: r.uNextBuffer, Offset: 0, Size: bufferSize},
			{Binding: 4, Buffer: r.vNextBuffer, Offset: 0, Size: bufferSize},
		},
	})
	if err != nil {
		return err
	}

	r.bindGroupAlt, err = r.device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Layout: r.bindGroupLayout,
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: r.paramsBuffer, Offset: 0, Size: 32},
			{Binding: 1, Buffer: r.uNextBuffer, Offset: 0, Size: bufferSize},
			{Binding: 2, Buffer: r.vNextBuffer, Offset: 0, Size: bufferSize},
			{Binding: 3, Buffer: r.uBuffer, Offset: 0, Size: bufferSize},
			{Binding: 4, Buffer: r.vBuffer, Offset: 0, Size: bufferSize},
		},
	})
	if err != nil {
		return err
	}

	r.width = width
	r.height = height
	r.bufferSize = bufferSize
	r.useAlt = false
	return nil
}

func (r *gpuRD) UploadState(width, height int, u, v []float64) error {
	if len(u) == 0 || len(u) != len(v) {
		return fmt.Errorf("invalid state buffer sizes")
	}
	if err := r.ensureBuffers(width, height); err != nil {
		return err
	}
	if len(r.uUpload) != len(u) {
		r.uUpload = make([]float32, len(u))
		r.vUpload = make([]float32, len(v))
	}
	for i := range u {
		r.uUpload[i] = float32(u[i])
		r.vUpload[i] = float32(v[i])
	}
	if len(r.uUpload) > 0 {
		r.queue.WriteBuffer(r.uBuffer, 0, float32Bytes(r.uUpload))
		r.queue.WriteBuffer(r.vBuffer, 0, float32Bytes(r.vUpload))
		r.queue.WriteBuffer(r.uNextBuffer, 0, float32Bytes(r.uUpload))
		r.queue.WriteBuffer(r.vNextBuffer, 0, float32Bytes(r.vUpload))
	}
	r.useAlt = false
	return nil
}

func (r *gpuRD) Step(f *rdField) error {
	if f.width != r.width || f.height != r.height || r.bindGroup == nil {
		if err := r.ensureBuffers(f.width, f.height); err != nil {
			return err
		}
	}
	params := gpuParams{
		Width:  uint32(f.width),
		Height: uint32(f.height),
		Du:     float32(f.du),
		Dv:     float32(f.dv),
		Feed:   float32(f.feed),
		Kill:   float32(f.kill),
		Dt:     float32(f.dt),
		Pad:    0,
	}
	paramBytes := encodeParams(params)
	r.queue.WriteBuffer(r.paramsBuffer, 0, paramBytes)

	encoder, err := r.device.CreateCommandEncoder(nil)
	if err != nil {
		return err
	}
	defer encoder.Release()

	pass := encoder.BeginComputePass(&wgpu.ComputePassDescriptor{Label: "reaction-diffusion-pass"})
	pass.SetPipeline(r.pipeline)
	if r.useAlt {
		pass.SetBindGroup(0, r.bindGroupAlt, nil)
	} else {
		pass.SetBindGroup(0, r.bindGroup, nil)
	}

	const workgroup = 8
	groupsX := uint32((f.width + workgroup - 1) / workgroup)
	groupsY := uint32((f.height + workgroup - 1) / workgroup)
	pass.DispatchWorkgroups(groupsX, groupsY, 1)
	if err := pass.End(); err != nil {
		pass.Release()
		return err
	}
	pass.Release()

	cmd, err := encoder.Finish(nil)
	if err != nil {
		return err
	}
	defer cmd.Release()
	r.queue.Submit(cmd)

	r.useAlt = !r.useAlt
	return nil
}

func (r *gpuRD) ReadbackV(f *rdField) error {
	if f.width == 0 || f.height == 0 {
		return nil
	}
	vBuffer := r.currentVBuffer()
	if vBuffer == nil {
		return fmt.Errorf("GPU buffers not ready")
	}
	data, err := r.readbackFloat32(vBuffer)
	if err != nil {
		return err
	}
	if len(f.v) != len(data) {
		f.v = make([]float64, len(data))
	}
	for i := range data {
		f.v[i] = float64(data[i])
	}
	return nil
}

func (r *gpuRD) InjectSeeds(f *rdField) error {
	if err := r.ReadbackState(f); err != nil {
		return err
	}
	for i := 0; i < f.injectCount; i++ {
		f.injectSeed()
	}
	return r.UploadState(f.width, f.height, f.u, f.v)
}

func (r *gpuRD) ReadbackState(f *rdField) error {
	if f.width == 0 || f.height == 0 {
		return nil
	}
	uBuffer := r.currentUBuffer()
	vBuffer := r.currentVBuffer()
	if uBuffer == nil || vBuffer == nil {
		return fmt.Errorf("GPU buffers not ready")
	}
	uData, err := r.readbackFloat32(uBuffer)
	if err != nil {
		return err
	}
	vData, err := r.readbackFloat32(vBuffer)
	if err != nil {
		return err
	}
	if len(f.u) != len(uData) {
		f.u = make([]float64, len(uData))
		f.v = make([]float64, len(vData))
	}
	for i := range uData {
		f.u[i] = float64(uData[i])
		f.v[i] = float64(vData[i])
	}
	return nil
}

func (r *gpuRD) readbackFloat32(src *wgpu.Buffer) ([]float32, error) {
	if src == nil || r.readbackBuffer == nil {
		return nil, fmt.Errorf("readback buffer not ready")
	}
	encoder, err := r.device.CreateCommandEncoder(nil)
	if err != nil {
		return nil, err
	}
	defer encoder.Release()
	encoder.CopyBufferToBuffer(src, 0, r.readbackBuffer, 0, r.bufferSize)
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
	mappedSlice := wgpu.FromBytes[float32](raw)
	data := make([]float32, len(mappedSlice))
	copy(data, mappedSlice)
	r.readbackBuffer.Unmap()
	return data, nil
}

func (r *gpuRD) currentUBuffer() *wgpu.Buffer {
	if r.useAlt {
		return r.uNextBuffer
	}
	return r.uBuffer
}

func (r *gpuRD) currentVBuffer() *wgpu.Buffer {
	if r.useAlt {
		return r.vNextBuffer
	}
	return r.vBuffer
}

func (r *gpuRD) Close() {
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
	if r.uNextBuffer != nil {
		r.uNextBuffer.Release()
		r.uNextBuffer = nil
	}
	if r.vNextBuffer != nil {
		r.vNextBuffer.Release()
		r.vNextBuffer = nil
	}
	if r.uBuffer != nil {
		r.uBuffer.Release()
		r.uBuffer = nil
	}
	if r.vBuffer != nil {
		r.vBuffer.Release()
		r.vBuffer = nil
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

func encodeParams(p gpuParams) []byte {
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

func paletteStops(p paletteType) []paletteStop {
	switch p {
	case paletteFire:
		return []paletteStop{
			{t: 0.0, c: color.RGBA{R: 6, G: 2, B: 8, A: 255}},
			{t: 0.35, c: color.RGBA{R: 120, G: 10, B: 24, A: 255}},
			{t: 0.7, c: color.RGBA{R: 230, G: 70, B: 20, A: 255}},
			{t: 1.0, c: color.RGBA{R: 245, G: 210, B: 80, A: 255}},
		}
	case paletteIce:
		return []paletteStop{
			{t: 0.0, c: color.RGBA{R: 6, G: 12, B: 22, A: 255}},
			{t: 0.4, c: color.RGBA{R: 20, G: 90, B: 130, A: 255}},
			{t: 0.75, c: color.RGBA{R: 120, G: 220, B: 235, A: 255}},
			{t: 1.0, c: color.RGBA{R: 230, G: 250, B: 255, A: 255}},
		}
	case paletteForest:
		return []paletteStop{
			{t: 0.0, c: color.RGBA{R: 5, G: 12, B: 8, A: 255}},
			{t: 0.35, c: color.RGBA{R: 16, G: 70, B: 40, A: 255}},
			{t: 0.7, c: color.RGBA{R: 70, G: 150, B: 90, A: 255}},
			{t: 1.0, c: color.RGBA{R: 190, G: 230, B: 160, A: 255}},
		}
	case paletteMono:
		return []paletteStop{
			{t: 0.0, c: color.RGBA{R: 4, G: 6, B: 8, A: 255}},
			{t: 1.0, c: color.RGBA{R: 235, G: 235, B: 235, A: 255}},
		}
	case paletteViridis:
		return []paletteStop{
			{t: 0.0, c: color.RGBA{R: 7, G: 16, B: 49, A: 255}},
			{t: 0.25, c: color.RGBA{R: 45, G: 70, B: 115, A: 255}},
			{t: 0.55, c: color.RGBA{R: 66, G: 135, B: 118, A: 255}},
			{t: 0.85, c: color.RGBA{R: 178, G: 214, B: 96, A: 255}},
			{t: 1.0, c: color.RGBA{R: 245, G: 235, B: 115, A: 255}},
		}
	default:
		return []paletteStop{
			{t: 0.0, c: color.RGBA{R: 6, G: 10, B: 18, A: 255}},
			{t: 0.35, c: color.RGBA{R: 48, G: 40, B: 90, A: 255}},
			{t: 0.7, c: color.RGBA{R: 160, G: 70, B: 130, A: 255}},
			{t: 1.0, c: color.RGBA{R: 250, G: 190, B: 120, A: 255}},
		}
	}
}

func colorFromPalette(p paletteType, t float64) color.RGBA {
	if t <= 0 {
		return paletteStops(p)[0].c
	}
	if t >= 1 {
		stops := paletteStops(p)
		return stops[len(stops)-1].c
	}
	stops := paletteStops(p)
	for i := 0; i < len(stops)-1; i++ {
		if t >= stops[i].t && t <= stops[i+1].t {
			dt := (t - stops[i].t) / (stops[i+1].t - stops[i].t)
			return lerpColor(stops[i].c, stops[i+1].c, dt)
		}
	}
	return stops[len(stops)-1].c
}

func lerpColor(a, b color.RGBA, t float64) color.RGBA {
	return color.RGBA{
		R: uint8(float64(a.R) + (float64(b.R)-float64(a.R))*t),
		G: uint8(float64(a.G) + (float64(b.G)-float64(a.G))*t),
		B: uint8(float64(a.B) + (float64(b.B)-float64(a.B))*t),
		A: 255,
	}
}

func drawField(img *image.RGBA, f *rdField, palette paletteType, cellSize int) {
	w, h := f.width, f.height
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			idx := y*w + x
			v := f.v[idx]
			u := f.u[idx]
			xm1 := x - 1
			if xm1 < 0 {
				xm1 = w - 1
			}
			xp1 := x + 1
			if xp1 == w {
				xp1 = 0
			}
			ym1 := y - 1
			if ym1 < 0 {
				ym1 = h - 1
			}
			yp1 := y + 1
			if yp1 == h {
				yp1 = 0
			}

			vx := f.v[y*w+xp1] - f.v[y*w+xm1]
			vy := f.v[yp1*w+x] - f.v[ym1*w+x]
			edge := math.Sqrt(vx*vx+vy*vy) * 1.2

			mix := 0.72*v + 0.28*(1.0-u)
			mix = mix*1.45 + edge*0.55
			t := clamp01(math.Pow(mix, 0.56))
			c := colorFromPalette(palette, t)
			px0 := x * cellSize
			py0 := y * cellSize
			px1 := min(px0+cellSize, img.Rect.Dx())
			py1 := min(py0+cellSize, img.Rect.Dy())
			fillRect(img, px0, py0, px1, py1, c)
		}
	}
}

func fillRect(img *image.RGBA, x0, y0, x1, y1 int, c color.RGBA) {
	if x0 < 0 {
		x0 = 0
	}
	if y0 < 0 {
		y0 = 0
	}
	if x1 > img.Rect.Dx() {
		x1 = img.Rect.Dx()
	}
	if y1 > img.Rect.Dy() {
		y1 = img.Rect.Dy()
	}
	if x0 >= x1 || y0 >= y1 {
		return
	}

	for y := y0; y < y1; y++ {
		offset := y*img.Stride + x0*4
		for x := x0; x < x1; x++ {
			img.Pix[offset] = c.R
			img.Pix[offset+1] = c.G
			img.Pix[offset+2] = c.B
			img.Pix[offset+3] = 255
			offset += 4
		}
	}
}

func (f *rdField) Resize(w, h int) {
	if w < 4 {
		w = 4
	}
	if h < 4 {
		h = 4
	}
	f.width = w
	f.height = h
	f.u = make([]float64, w*h)
	f.v = make([]float64, w*h)
	f.uNext = make([]float64, w*h)
	f.vNext = make([]float64, w*h)
	f.seedMinRadius = max(2, w/80)
	f.seedMaxRadius = max(f.seedMinRadius+1, w/26)
	f.seedCount = max(4, (w*h)/18000)
	f.seedEdgeJitter = 0.15
	if f.gpu != nil {
		if err := f.gpu.ensureBuffers(w, h); err != nil {
			fmt.Fprintf(os.Stderr, "GPU buffer allocation failed: %v\n", err)
			f.gpu.Close()
			f.gpu = nil
		}
	}
}

func (f *rdField) Reset() {
	for i := range f.u {
		f.u[i] = 1.0
		f.v[i] = 0.0
	}
	f.step = 0
	f.stepsSinceReset = 0
	f.resetHitCount = 0
	for i := 0; i < f.seedCount; i++ {
		f.injectSeed()
	}
	if f.gpu != nil {
		if err := f.gpu.UploadState(f.width, f.height, f.u, f.v); err != nil {
			fmt.Fprintf(os.Stderr, "GPU upload failed: %v\n", err)
			f.gpu.Close()
			f.gpu = nil
		}
	}
}

func (f *rdField) Step() {
	if f.gpu != nil {
		if err := f.gpu.Step(f); err != nil {
			fmt.Fprintf(os.Stderr, "GPU step failed: %v\n", err)
			f.gpu.Close()
			f.gpu = nil
		} else {
			f.step++
			f.stepsSinceReset++
			if f.injectEvery > 0 && f.step%f.injectEvery == 0 {
				if err := f.gpu.InjectSeeds(f); err != nil {
					fmt.Fprintf(os.Stderr, "GPU seed inject failed: %v\n", err)
				}
			}
			return
		}
	}

	w := f.width
	h := f.height
	for y := 0; y < h; y++ {
		ym1 := y - 1
		if ym1 < 0 {
			ym1 = h - 1
		}
		yp1 := y + 1
		if yp1 == h {
			yp1 = 0
		}

		for x := 0; x < w; x++ {
			xm1 := x - 1
			if xm1 < 0 {
				xm1 = w - 1
			}
			xp1 := x + 1

			idx := y*w + x
			u := f.u[idx]
			v := f.v[idx]

			lapU := -u
			lapV := -v

			lapU += (f.u[y*w+xm1] + f.u[y*w+xp1] + f.u[ym1*w+x] + f.u[yp1*w+x]) * 0.2
			lapV += (f.v[y*w+xm1] + f.v[y*w+xp1] + f.v[ym1*w+x] + f.v[yp1*w+x]) * 0.2

			lapU += (f.u[ym1*w+xm1] + f.u[ym1*w+xp1] + f.u[yp1*w+xm1] + f.u[yp1*w+xp1]) * 0.05
			lapV += (f.v[ym1*w+xm1] + f.v[ym1*w+xp1] + f.v[yp1*w+xm1] + f.v[yp1*w+xp1]) * 0.05

			uvv := u * v * v
			du := f.du*lapU - uvv + f.feed*(1.0-u)
			dv := f.dv*lapV + uvv - (f.kill+f.feed)*v

			f.uNext[idx] = clamp01(u + du*f.dt)
			f.vNext[idx] = clamp01(v + dv*f.dt)
		}
	}

	f.u, f.uNext = f.uNext, f.u
	f.v, f.vNext = f.vNext, f.v
	f.step++
	f.stepsSinceReset++

	if f.injectEvery > 0 && f.step%f.injectEvery == 0 {
		for i := 0; i < f.injectCount; i++ {
			f.injectSeed()
		}
	}
}

func (f *rdField) injectSeed() {
	if f.width == 0 || f.height == 0 {
		return
	}
	cx := f.rnd.Intn(f.width)
	cy := f.rnd.Intn(f.height)
	radius := f.seedMinRadius
	if f.seedMaxRadius > f.seedMinRadius {
		radius = f.seedMinRadius + f.rnd.Intn(f.seedMaxRadius-f.seedMinRadius+1)
	}

	for y := -radius; y <= radius; y++ {
		py := (cy + y + f.height) % f.height
		dy := float64(y) / float64(radius)
		for x := -radius; x <= radius; x++ {
			px := (cx + x + f.width) % f.width
			dx := float64(x) / float64(radius)
			if dx*dx+dy*dy <= 1.0+f.seedEdgeJitter*f.rnd.Float64() {
				idx := py*f.width + px
				f.v[idx] = 0.35 + 0.25*f.rnd.Float64()
				f.u[idx] = 0.55 + 0.25*f.rnd.Float64()
			}
		}
	}
}

func (f *rdField) shouldReset(viewW, viewH int) bool {
	if f.width == 0 || f.height == 0 {
		return false
	}
	if viewW <= 0 || viewH <= 0 {
		return false
	}
	if viewW > f.width {
		viewW = f.width
	}
	if viewH > f.height {
		viewH = f.height
	}
	if f.stepsSinceReset < f.resetMinSteps {
		return false
	}
	if f.resetCooldown > 0 && f.stepsSinceReset < f.resetCooldown {
		return false
	}
	if f.resetMaxSteps > 0 && f.stepsSinceReset >= f.resetMaxSteps {
		return true
	}
	stride := f.resetStride
	if stride < 1 {
		stride = 1
	}
	count := 0
	full := 0
	mean := 0.0
	mean2 := 0.0
	for y := 0; y < viewH; y += stride {
		row := y * f.width
		for x := 0; x < viewW; x += stride {
			v := f.v[row+x]
			u := f.u[row+x]
			mix := 0.72*v + 0.28*(1.0-u)
			mix *= 1.25
			t := clamp01(math.Pow(mix, 0.6))
			if t >= f.resetThreshold {
				full++
			}
			mean += v
			mean2 += v * v
			count++
		}
	}
	if count == 0 {
		return false
	}
	mean /= float64(count)
	mean2 /= float64(count)
	variance := mean2 - mean*mean

	trigger := false
	if float64(full)/float64(count) >= f.resetFraction {
		trigger = true
	}
	if mean > 0.18 && variance < 0.0011 {
		trigger = true
	}
	if trigger {
		f.resetHitCount++
	} else if f.resetHitCount > 0 {
		f.resetHitCount--
	}
	if f.resetHitNeeded < 1 {
		f.resetHitNeeded = 1
	}
	if f.resetHitCount >= f.resetHitNeeded {
		f.resetHitCount = 0
		f.lastResetStep = f.step
		return true
	}
	return false
}

func clamp01(v float64) float64 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
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

func getTermPixels() (int, int) {
	ws := &Winsize{}
	_, _, err := syscall.Syscall(syscall.SYS_IOCTL, uintptr(syscall.Stdout), uintptr(syscall.TIOCGWINSZ), uintptr(unsafe.Pointer(ws)))
	w, h := int(ws.Xpixel), int(ws.Ypixel)
	if err != 0 || w == 0 || h == 0 {
		w, h = int(ws.Col)*10, int(ws.Row)*20
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

const reactionDiffusionWGSL = `
struct Params {
	width: u32,
	height: u32,
	du: f32,
	dv: f32,
	feed: f32,
	kill: f32,
	dt: f32,
	pad: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> uIn: array<f32>;
@group(0) @binding(2) var<storage, read> vIn: array<f32>;
@group(0) @binding(3) var<storage, read_write> uOut: array<f32>;
@group(0) @binding(4) var<storage, read_write> vOut: array<f32>;

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
	let u = uIn[i];
	let v = vIn[i];

	var lapU = -u;
	var lapV = -v;

	lapU += 0.2 * (uIn[idx(xm1, y, w)] + uIn[idx(xp1, y, w)] + uIn[idx(x, ym1, w)] + uIn[idx(x, yp1, w)]);
	lapV += 0.2 * (vIn[idx(xm1, y, w)] + vIn[idx(xp1, y, w)] + vIn[idx(x, ym1, w)] + vIn[idx(x, yp1, w)]);

	lapU += 0.05 * (uIn[idx(xm1, ym1, w)] + uIn[idx(xp1, ym1, w)] + uIn[idx(xm1, yp1, w)] + uIn[idx(xp1, yp1, w)]);
	lapV += 0.05 * (vIn[idx(xm1, ym1, w)] + vIn[idx(xp1, ym1, w)] + vIn[idx(xm1, yp1, w)] + vIn[idx(xp1, yp1, w)]);

	let uvv = u * v * v;
	let du = params.du * lapU - uvv + params.feed * (1.0 - u);
	let dv = params.dv * lapV + uvv - (params.kill + params.feed) * v;

	uOut[i] = clamp(u + du * params.dt, 0.0, 1.0);
	vOut[i] = clamp(v + dv * params.dt, 0.0, 1.0);
}
`
