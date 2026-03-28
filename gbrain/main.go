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
	"syscall"
	"time"
	"unsafe"

	"github.com/okieraised/gonii"
	"github.com/rajveermalviya/go-webgpu/wgpu"
)

type Winsize struct {
	Row    uint16
	Col    uint16
	Xpixel uint16
	Ypixel uint16
}

type vec3 struct{ x, y, z float64 }

type paletteType int
type colorModeType int
type renderEngine int

type gpuVolumeRenderer struct {
	instance        *wgpu.Instance
	adapter         *wgpu.Adapter
	device          *wgpu.Device
	queue           *wgpu.Queue
	pipeline        *wgpu.ComputePipeline
	bindGroupLayout *wgpu.BindGroupLayout
	paramsBuffer    *wgpu.Buffer
	volumeBuffer    *wgpu.Buffer
	outputBuffer    *wgpu.Buffer
	readbackBuffer  *wgpu.Buffer
	bindGroup       *wgpu.BindGroup
	bufferSize      uint64
	width           int
	height          int
	volNx           int
	volNy           int
	volNz           int
}

type gpuFrameParams struct {
	Yaw       float32
	Pitch     float32
	Roll      float32
	FitMargin float32
	Opacity   float32
	Iso       float32
	EdgeBoost float32
	Samples   uint32
	PaletteDensity uint32
	PaletteEdge    uint32
	ColorMode uint32
}

const (
	paletteTwilight paletteType = iota
	paletteFire
	paletteIce
	paletteForest
	paletteMono
)

const (
	colorModeDensity colorModeType = iota
	colorModeEdge
	colorModeDensityEdge
	colorModeDepth
	colorModeNormal
	colorModeOpacity
)

const (
	renderEngineAuto renderEngine = iota
	renderEngineCPU
	renderEngineGPU
)

type volume struct {
	nx, ny, nz int
	data       []uint8
}

var framePNGEncoder = png.Encoder{CompressionLevel: png.BestSpeed}
var frameBuffer bytes.Buffer
var frameBase64Buffer []byte

func main() {
	niftiPath := flag.String("nii", "./average305_t1_tal_lin.nii", "Path to NIfTI file (.nii or .nii.gz)")
	spm := flag.Int("spm", 90, "Simulation steps per minute")
	rotationSpeed := flag.Float64("rotation-speed", 0.22, "Camera orbit speed in radians per second")
	zoom := flag.Float64("zoom", 1.0, "Camera zoom (1.0 fits whole model, >1 zooms in, <1 zooms out)")
	opacity := flag.Float64("opacity", 0.24, "Base volume opacity (0..2)")
	renderScale := flag.Float64("render-scale", 0.82, "Internal render scale (0.2..1.0)")
	samples := flag.Int("samples", 240, "Maximum ray-marching samples per ray")
	maxDim := flag.Int("max-dim", 180, "Maximum loaded volume dimension (downsampled for speed)")
	paletteName := flag.String("palette", "twilight", "Color palette: twilight|fire|ice|forest|mono")
	paletteDensityName := flag.String("palette-density", "", "Density palette override (defaults to -palette)")
	paletteEdgeName := flag.String("palette-edge", "fire", "Edge palette used by -color-mode=density-edge")
	colorModeName := flag.String("color-mode", "density", "Color mode: density|edge|density-edge|depth|normal|opacity")
	engineName := flag.String("engine", "auto", "Render engine: auto|cpu|gpu")
	iso := flag.Float64("iso", 0.13, "Soft density threshold for tissue visibility (0..1)")
	edgeBoost := flag.Float64("edge-boost", 1.8, "Edge/detail boost multiplier")
	tiltMaxDeg := flag.Float64("tilt-max", 24, "Maximum random tilt angle in degrees")
	angleChangeSec := flag.Float64("angle-change-sec", 14, "Seconds between random tilt targets")
	frameStride := flag.Int("frame-stride", 1, "Render one frame every N simulation steps")
	flag.Parse()

	if *spm < 1 {
		*spm = 1
	}
	if *renderScale < 0.2 {
		*renderScale = 0.2
	}
	if *renderScale > 1 {
		*renderScale = 1
	}
	if *samples < 32 {
		*samples = 32
	}
	if *samples > 420 {
		*samples = 420
	}
	if *maxDim < 48 {
		*maxDim = 48
	}
	if *maxDim > 256 {
		*maxDim = 256
	}
	if *tiltMaxDeg < 0 {
		*tiltMaxDeg = 0
	}
	if *tiltMaxDeg > 65 {
		*tiltMaxDeg = 65
	}
	if *angleChangeSec < 2 {
		*angleChangeSec = 2
	}
	if *opacity < 0.02 {
		*opacity = 0.02
	}
	if *opacity > 2 {
		*opacity = 2
	}
	if *zoom <= 0 {
		*zoom = 1.0
	}
	if *zoom < 0.2 {
		*zoom = 0.2
	}
	if *zoom > 8 {
		*zoom = 8
	}
	if *iso < 0 {
		*iso = 0
	}
	if *iso > 0.8 {
		*iso = 0.8
	}
	if *edgeBoost < 0 {
		*edgeBoost = 0
	}
	if *edgeBoost > 5 {
		*edgeBoost = 5
	}
	if *frameStride < 1 {
		*frameStride = 1
	}

	fitMargin := 1.10 / *zoom

	paletteDensity := parsePalette(*paletteName)
	if normalizeName(*paletteDensityName) != "" {
		paletteDensity = parsePalette(*paletteDensityName)
	}
	paletteEdge := parsePalette(*paletteEdgeName)
	colorMode := parseColorMode(*colorModeName)
	engine := parseRenderEngine(*engineName)

	fmt.Fprintf(os.Stderr, "loading NIfTI volume: %s\n", *niftiPath)
	vol, err := loadVolume(*niftiPath, *maxDim)
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to load NIfTI: %v\n", err)
		os.Exit(1)
	}
	fmt.Fprintf(os.Stderr, "volume ready: %dx%dx%d\n", vol.nx, vol.ny, vol.nz)

	var gpu *gpuVolumeRenderer
	if engine != renderEngineCPU {
		r, err := newGPUVolumeRenderer(vol)
		if err != nil {
			if engine == renderEngineGPU {
				fmt.Fprintf(os.Stderr, "failed to initialize GPU renderer: %v\n", err)
				os.Exit(1)
			}
			fmt.Fprintf(os.Stderr, "GPU renderer unavailable (%v), falling back to CPU\n", err)
		} else {
			gpu = r
			fmt.Fprintln(os.Stderr, "using WebGPU volume renderer")
			defer gpu.Close()
		}
	}

	rnd := rand.New(rand.NewSource(time.Now().UnixNano()))
	tiltLimit := *tiltMaxDeg * math.Pi / 180
	yaw := rnd.Float64() * 2 * math.Pi
	pitch := (rnd.Float64()*2 - 1) * tiltLimit
	roll := (rnd.Float64()*2 - 1) * (tiltLimit * 0.4)
	targetYaw := yaw
	targetPitch := pitch
	targetRoll := roll
	rotationDir := 1.0
	if rnd.Float64() < 0.5 {
		rotationDir = -1.0
	}
	stepsPerAngleChange := int(math.Max(1, math.Round(*angleChangeSec*float64(*spm)/60)))

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

	ticker := time.NewTicker(time.Minute / time.Duration(*spm))
	defer ticker.Stop()
	resizeTicker := time.NewTicker(700 * time.Millisecond)
	defer resizeTicker.Stop()

	step := 0
	currentID := 1
	previousID := 2

	render := func() {
		w, h := getTermPixels()
		if w < 24 || h < 24 {
			return
		}

		rw := maxInt(64, int(float64(w)**renderScale))
		rh := maxInt(64, int(float64(h)**renderScale))

		low := image.NewRGBA(image.Rect(0, 0, rw, rh))
		high := image.NewRGBA(image.Rect(0, 0, w, h))

		if gpu != nil {
			gpuImg, err := gpu.Render(rw, rh, gpuFrameParams{
				Yaw:       float32(yaw),
				Pitch:     float32(pitch),
				Roll:      float32(roll),
				FitMargin: float32(fitMargin),
				Opacity:   float32(*opacity),
				Iso:       float32(*iso),
				EdgeBoost: float32(*edgeBoost),
				Samples:   uint32(*samples),
				PaletteDensity: uint32(paletteDensity),
				PaletteEdge:    uint32(paletteEdge),
				ColorMode: uint32(colorMode),
			})
			if err != nil {
				if engine == renderEngineGPU {
					fmt.Fprintf(os.Stderr, "GPU render failed: %v\n", err)
					os.Exit(1)
				}
				fmt.Fprintf(os.Stderr, "GPU render failed (%v), falling back to CPU\n", err)
				gpu.Close()
				gpu = nil
				renderVolume(low, vol, yaw, pitch, roll, fitMargin, *opacity, *samples, *iso, *edgeBoost, paletteDensity, paletteEdge, colorMode)
			} else {
				low = gpuImg
			}
		} else {
			renderVolume(low, vol, yaw, pitch, roll, fitMargin, *opacity, *samples, *iso, *edgeBoost, paletteDensity, paletteEdge, colorMode)
		}
		upscaleNearest(low, high)

		fmt.Print("\033[H")
		printKittyImage(high, currentID)
		fmt.Printf("\033_Ga=d,d=i,q=2,i=%d\033\\", previousID)
		currentID, previousID = previousID, currentID
	}

	render()
	for {
		select {
		case <-ticker.C:
			step++
			dt := 60.0 / float64(*spm)
			yaw = wrapAngle(yaw + (*rotationSpeed * rotationDir * dt))
			if step%stepsPerAngleChange == 0 {
				targetYaw = wrapAngle(yaw + (rnd.Float64()*2-1)*math.Pi*0.9)
				targetPitch = (rnd.Float64()*2 - 1) * tiltLimit
				targetRoll = (rnd.Float64()*2 - 1) * (tiltLimit * 0.4)
				if rnd.Float64() < 0.35 {
					rotationDir *= -1
				}
			}
			yaw = wrapAngle(yaw + shortestAngleDelta(yaw, targetYaw)*0.025)
			pitch += (targetPitch - pitch) * 0.06
			roll += (targetRoll - roll) * 0.04
			if step%*frameStride == 0 {
				render()
			}
		case <-resizeTicker.C:
			render()
		}
	}
}

func loadVolume(path string, maxDim int) (*volume, error) {
	rd, err := gonii.NewNiiReader(
		gonii.WithReadImageFile(path),
		gonii.WithReadRetainHeader(false),
	)
	if err != nil {
		return nil, err
	}
	if err := rd.Parse(); err != nil {
		return nil, err
	}

	img := rd.GetNiiData()
	shape := img.GetImgShape()
	sx, sy, sz := int(shape[0]), int(shape[1]), int(shape[2])
	if sx <= 0 || sy <= 0 || sz <= 0 {
		return nil, fmt.Errorf("invalid image shape: %v", shape)
	}

	origMax := maxInt(sx, maxInt(sy, sz))
	scale := 1.0
	if origMax > maxDim {
		scale = float64(maxDim) / float64(origMax)
	}
	nx := maxInt(24, int(math.Round(float64(sx)*scale)))
	ny := maxInt(24, int(math.Round(float64(sy)*scale)))
	nz := maxInt(24, int(math.Round(float64(sz)*scale)))

	tmp := make([]float64, nx*ny*nz)
	mn := math.MaxFloat64
	mx := -math.MaxFloat64

	for z := 0; z < nz; z++ {
		fz := ((float64(z)+0.5)/float64(nz))*float64(sz) - 0.5
		for y := 0; y < ny; y++ {
			fy := ((float64(y)+0.5)/float64(ny))*float64(sy) - 0.5
			for x := 0; x < nx; x++ {
				fx := ((float64(x)+0.5)/float64(nx))*float64(sx) - 0.5
				v := sampleNiiTrilinear(img, fx, fy, fz, sx, sy, sz)
				i := ((z * ny) + y) * nx + x
				tmp[i] = v
				if v < mn {
					mn = v
				}
				if v > mx {
					mx = v
				}
			}
		}
	}

	if mx-mn <= 1e-9 {
		return nil, fmt.Errorf("empty or constant volume")
	}

	windowLow := mn + 0.04*(mx-mn)
	windowHigh := mn + 0.52*(mx-mn)
	if windowHigh <= windowLow {
		windowLow = mn
		windowHigh = mx
	}
	invWindow := 1.0 / (windowHigh - windowLow)

	out := make([]uint8, len(tmp))
	for i, v := range tmp {
		t := (v - windowLow) * invWindow
		if t < 0 {
			t = 0
		}
		if t > 1 {
			t = 1
		}
		t = math.Pow(t, 0.85)
		out[i] = uint8(t*255 + 0.5)
	}

	return &volume{nx: nx, ny: ny, nz: nz, data: out}, nil
}

func sampleNiiTrilinear(img interface{ GetAt(x, y, z, t int64) float64 }, x, y, z float64, sx, sy, sz int) float64 {
	x = clampf(x, 0, float64(sx-1))
	y = clampf(y, 0, float64(sy-1))
	z = clampf(z, 0, float64(sz-1))

	x0 := int(math.Floor(x))
	y0 := int(math.Floor(y))
	z0 := int(math.Floor(z))
	x1 := minInt(x0+1, sx-1)
	y1 := minInt(y0+1, sy-1)
	z1 := minInt(z0+1, sz-1)

	tx := x - float64(x0)
	ty := y - float64(y0)
	tz := z - float64(z0)

	c000 := img.GetAt(int64(x0), int64(y0), int64(z0), 0)
	c100 := img.GetAt(int64(x1), int64(y0), int64(z0), 0)
	c010 := img.GetAt(int64(x0), int64(y1), int64(z0), 0)
	c110 := img.GetAt(int64(x1), int64(y1), int64(z0), 0)
	c001 := img.GetAt(int64(x0), int64(y0), int64(z1), 0)
	c101 := img.GetAt(int64(x1), int64(y0), int64(z1), 0)
	c011 := img.GetAt(int64(x0), int64(y1), int64(z1), 0)
	c111 := img.GetAt(int64(x1), int64(y1), int64(z1), 0)

	c00 := c000*(1-tx) + c100*tx
	c10 := c010*(1-tx) + c110*tx
	c01 := c001*(1-tx) + c101*tx
	c11 := c011*(1-tx) + c111*tx
	c0 := c00*(1-ty) + c10*ty
	c1 := c01*(1-ty) + c11*ty
	return c0*(1-tz) + c1*tz
}

func renderVolume(img *image.RGBA, vol *volume, yaw, pitch, roll, fitMargin, opacity float64, samples int, iso, edgeBoost float64, paletteDensity, paletteEdge paletteType, colorMode colorModeType) {
	w := img.Rect.Dx()
	h := img.Rect.Dy()
	fillBackground(img, color.RGBA{R: 3, G: 7, B: 12, A: 255})

	fovY := 34.0 * math.Pi / 180.0
	aspect := float64(w) / float64(h)
	radius := fitCameraDistanceForUnitCube(fovY, aspect, fitMargin)

	cam := vec3{
		x: radius * math.Cos(pitch) * math.Cos(yaw),
		y: radius * math.Sin(pitch),
		z: radius * math.Cos(pitch) * math.Sin(yaw),
	}
	center := vec3{0, 0, 0}
	forward := norm(sub(center, cam))
	upWorld := vec3{0, 1, 0}
	if math.Abs(dot(forward, upWorld)) > 0.98 {
		upWorld = vec3{0, 0, 1}
	}
	right := norm(cross(forward, upWorld))
	up := norm(cross(right, forward))

	if roll != 0 {
		cr := math.Cos(roll)
		sr := math.Sin(roll)
		r2 := add(scale(right, cr), scale(up, sr))
		u2 := add(scale(up, cr), scale(right, -sr))
		right, up = r2, u2
	}

	tanHalf := math.Tan(fovY * 0.5)

	maxPath := math.Sqrt(3)
	stepSize := maxPath / float64(samples)
	baseAlpha := opacity * 0.66
	gradStep := 1.35 / float64(maxInt(vol.nx, maxInt(vol.ny, vol.nz)))
	lightDir := norm(add(forward, add(scale(right, -0.20), scale(up, 0.26))))

	for py := 0; py < h; py++ {
		for px := 0; px < w; px++ {
			sx := ((float64(px)+0.5)/float64(w))*2 - 1
			sy := 1 - ((float64(py)+0.5)/float64(h))*2
			sx *= aspect * tanHalf
			sy *= tanHalf

			dir := norm(add(forward, add(scale(right, sx), scale(up, sy))))

			t0, t1, ok := intersectUnitBox(cam, dir)
			if !ok {
				continue
			}
			if t0 < 0 {
				t0 = 0
			}

			accR, accG, accB, accA := 0.0, 0.0, 0.0, 0.0
			prev := 0.0
			rangeLen := math.Max(t1-t0, 1e-9)
			for t := t0; t < t1; t += stepSize {
				p := add(cam, scale(dir, t))
				u := p.x + 0.5
				v := p.y + 0.5
				wz := p.z + 0.5
				d := sampleVolumeTrilinear(vol, u, v, wz)
				if d <= iso*0.45 {
					prev = d
					continue
				}

				grad := sampleVolumeGradient(vol, u, v, wz, gradStep)
				gmag := length3(grad)
				edge := clampf(gmag*edgeBoost, 0, 1)

				delta := d - prev
				if delta < 0 {
					delta = 0
				}
				prev = d

				local := clampf((d-iso)/(1.0-iso), 0, 1)
				n := norm(grad)
				diff := dot(n, lightDir)
				if diff < 0 {
					diff = 0
				}
				shade := 0.32 + 0.68*diff

				a := baseAlpha * (0.10 + 0.90*local)
				a *= 0.42 + edge*0.92
				a *= 1.0 + delta*1.45
				if a > 0.9 {
					a = 0.9
				}

				depth := clampf((t-t0)/rangeLen, 0, 1)
				pr, pg, pb := colorFromMode(paletteDensity, paletteEdge, colorMode, local, edge, depth, n, a)
				colourBoost := 0.78 + 0.38*edge
				shadeFactor := shade
				if colorMode == colorModeNormal {
					shadeFactor = 1.0
				}
				cr := clampf(pr*shadeFactor*colourBoost, 0, 1)
				cg := clampf(pg*shadeFactor*colourBoost, 0, 1)
				cb := clampf(pb*shadeFactor*colourBoost, 0, 1)

				oneMinus := 1 - accA
				accR += oneMinus * a * cr
				accG += oneMinus * a * cg
				accB += oneMinus * a * cb
				accA += oneMinus * a
				if accA > 0.965 {
					break
				}
			}

			if accA <= 0 {
				continue
			}

			bg := color.RGBA{R: 3, G: 7, B: 12, A: 255}
			inv := 1 - accA
			r := uint8(clampf(accR*255+float64(bg.R)*inv, 0, 255))
			g := uint8(clampf(accG*255+float64(bg.G)*inv, 0, 255))
			b := uint8(clampf(accB*255+float64(bg.B)*inv, 0, 255))
			img.SetRGBA(px, py, color.RGBA{R: r, G: g, B: b, A: 255})
		}
	}
}

func sampleVolumeTrilinear(v *volume, u, w, z float64) float64 {
	if u < 0 || u > 1 || w < 0 || w > 1 || z < 0 || z > 1 {
		return 0
	}
	x := u * float64(v.nx-1)
	y := w * float64(v.ny-1)
	zz := z * float64(v.nz-1)

	x0 := int(x)
	y0 := int(y)
	z0 := int(zz)
	x1 := minInt(x0+1, v.nx-1)
	y1 := minInt(y0+1, v.ny-1)
	z1 := minInt(z0+1, v.nz-1)

	tx := x - float64(x0)
	ty := y - float64(y0)
	tz := zz - float64(z0)

	idx := func(ix, iy, iz int) int { return ((iz * v.ny) + iy) * v.nx + ix }
	g := func(ix, iy, iz int) float64 { return float64(v.data[idx(ix, iy, iz)]) / 255.0 }

	c000 := g(x0, y0, z0)
	c100 := g(x1, y0, z0)
	c010 := g(x0, y1, z0)
	c110 := g(x1, y1, z0)
	c001 := g(x0, y0, z1)
	c101 := g(x1, y0, z1)
	c011 := g(x0, y1, z1)
	c111 := g(x1, y1, z1)

	c00 := c000*(1-tx) + c100*tx
	c10 := c010*(1-tx) + c110*tx
	c01 := c001*(1-tx) + c101*tx
	c11 := c011*(1-tx) + c111*tx
	c0 := c00*(1-ty) + c10*ty
	c1 := c01*(1-ty) + c11*ty
	return c0*(1-tz) + c1*tz
}

func sampleVolumeGradient(v *volume, u, w, z, h float64) vec3 {
	if h <= 0 {
		h = 1.0 / float64(maxInt(v.nx, maxInt(v.ny, v.nz)))
	}
	x0 := sampleVolumeTrilinear(v, u-h, w, z)
	x1 := sampleVolumeTrilinear(v, u+h, w, z)
	y0 := sampleVolumeTrilinear(v, u, w-h, z)
	y1 := sampleVolumeTrilinear(v, u, w+h, z)
	z0 := sampleVolumeTrilinear(v, u, w, z-h)
	z1 := sampleVolumeTrilinear(v, u, w, z+h)
	inv := 0.5 / h
	return vec3{x: (x1 - x0) * inv, y: (y1 - y0) * inv, z: (z1 - z0) * inv}
}

func parsePalette(s string) paletteType {
	switch normalizeName(s) {
	case "fire":
		return paletteFire
	case "ice":
		return paletteIce
	case "forest":
		return paletteForest
	case "mono", "monochrome", "grayscale", "grey", "gray":
		return paletteMono
	default:
		return paletteTwilight
	}
}

func parseRenderEngine(s string) renderEngine {
	switch normalizeName(s) {
	case "cpu":
		return renderEngineCPU
	case "gpu", "webgpu", "wgpu":
		return renderEngineGPU
	default:
		return renderEngineAuto
	}
}

func parseColorMode(s string) colorModeType {
	switch normalizeName(s) {
	case "edge":
		return colorModeEdge
	case "densityedge", "density-edge":
		return colorModeDensityEdge
	case "depth":
		return colorModeDepth
	case "normal", "normals":
		return colorModeNormal
	case "opacity", "alpha":
		return colorModeOpacity
	default:
		return colorModeDensity
	}
}

func normalizeName(s string) string {
	b := make([]rune, 0, len(s))
	for _, r := range s {
		if r >= 'A' && r <= 'Z' {
			r = r - 'A' + 'a'
		}
		if r == ' ' || r == '_' {
			continue
		}
		b = append(b, r)
	}
	return string(b)
}

func colorFromPalette(p paletteType, t float64) (float64, float64, float64) {
	t = clampf(t, 0, 1)
	switch p {
	case paletteFire:
		return gradient3(t,
			0.02, 0.01, 0.04,
			0.75, 0.12, 0.03,
			1.00, 0.88, 0.18,
		)
	case paletteIce:
		return gradient3(t,
			0.01, 0.06, 0.10,
			0.08, 0.44, 0.70,
			0.80, 0.97, 1.00,
		)
	case paletteForest:
		return gradient3(t,
			0.01, 0.05, 0.02,
			0.08, 0.45, 0.16,
			0.86, 0.97, 0.78,
		)
	case paletteMono:
		g := 0.08 + 0.92*t
		return g, g, g
	default: // twilight
		return gradient3(t,
			0.06, 0.03, 0.14,
			0.42, 0.22, 0.70,
			0.97, 0.80, 0.98,
		)
	}
}

func colorFromMode(pDensity, pEdge paletteType, mode colorModeType, local, edge, depth float64, n vec3, alpha float64) (float64, float64, float64) {
	switch mode {
	case colorModeEdge:
		return colorFromPalette(pEdge, clampf(edge, 0, 1))
	case colorModeDensityEdge:
		dr, dg, db := colorFromPalette(pDensity, clampf(local, 0, 1))
		er, eg, eb := colorFromPalette(pEdge, clampf(edge, 0, 1))
		mixE := clampf(0.42*edge+0.18, 0, 0.78)
		return mix(dr, er, mixE), mix(dg, eg, mixE), mix(db, eb, mixE)
	case colorModeDepth:
		return colorFromPalette(pDensity, 1.0-clampf(depth, 0, 1))
	case colorModeNormal:
		return clampf(0.5*(n.x+1), 0, 1), clampf(0.5*(n.y+1), 0, 1), clampf(0.5*(n.z+1), 0, 1)
	case colorModeOpacity:
		return colorFromPalette(pDensity, clampf(alpha/0.9, 0, 1))
	default:
		return colorFromPalette(pDensity, local)
	}
}

func gradient3(t, r0, g0, b0, r1, g1, b1, r2, g2, b2 float64) (float64, float64, float64) {
	if t < 0.5 {
		u := t * 2
		return mix(r0, r1, u), mix(g0, g1, u), mix(b0, b1, u)
	}
	u := (t - 0.5) * 2
	return mix(r1, r2, u), mix(g1, g2, u), mix(b1, b2, u)
}

func mix(a, b, t float64) float64 { return a + (b-a)*t }

func intersectUnitBox(origin, dir vec3) (float64, float64, bool) {
	const eps = 1e-9
	boxMin := vec3{-0.5, -0.5, -0.5}
	boxMax := vec3{0.5, 0.5, 0.5}

	tMin := -math.MaxFloat64
	tMax := math.MaxFloat64

	axis := func(o, d, mn, mx float64) (float64, float64, bool) {
		if math.Abs(d) < eps {
			if o < mn || o > mx {
				return 0, 0, false
			}
			return -math.MaxFloat64, math.MaxFloat64, true
		}
		t0 := (mn - o) / d
		t1 := (mx - o) / d
		if t0 > t1 {
			t0, t1 = t1, t0
		}
		return t0, t1, true
	}

	a0, a1, ok := axis(origin.x, dir.x, boxMin.x, boxMax.x)
	if !ok {
		return 0, 0, false
	}
	tMin = math.Max(tMin, a0)
	tMax = math.Min(tMax, a1)

	a0, a1, ok = axis(origin.y, dir.y, boxMin.y, boxMax.y)
	if !ok {
		return 0, 0, false
	}
	tMin = math.Max(tMin, a0)
	tMax = math.Min(tMax, a1)

	a0, a1, ok = axis(origin.z, dir.z, boxMin.z, boxMax.z)
	if !ok {
		return 0, 0, false
	}
	tMin = math.Max(tMin, a0)
	tMax = math.Min(tMax, a1)

	if tMax < tMin {
		return 0, 0, false
	}
	return tMin, tMax, true
}

func upscaleNearest(src, dst *image.RGBA) {
	sw := src.Rect.Dx()
	sh := src.Rect.Dy()
	dw := dst.Rect.Dx()
	dh := dst.Rect.Dy()

	for y := 0; y < dh; y++ {
		sy := (y * sh) / dh
		for x := 0; x < dw; x++ {
			sx := (x * sw) / dw
			offS := src.PixOffset(sx, sy)
			offD := dst.PixOffset(x, y)
			dst.Pix[offD+0] = src.Pix[offS+0]
			dst.Pix[offD+1] = src.Pix[offS+1]
			dst.Pix[offD+2] = src.Pix[offS+2]
			dst.Pix[offD+3] = 255
		}
	}
}

func fillBackground(img *image.RGBA, c color.RGBA) {
	for i := 0; i < len(img.Pix); i += 4 {
		img.Pix[i+0] = c.R
		img.Pix[i+1] = c.G
		img.Pix[i+2] = c.B
		img.Pix[i+3] = c.A
	}
}

func printKittyImage(img image.Image, id int) {
	frameBuffer.Reset()
	_ = framePNGEncoder.Encode(&frameBuffer, img)
	encodedLen := base64.StdEncoding.EncodedLen(frameBuffer.Len())
	if cap(frameBase64Buffer) < encodedLen {
		frameBase64Buffer = make([]byte, encodedLen)
	} else {
		frameBase64Buffer = frameBase64Buffer[:encodedLen]
	}
	base64.StdEncoding.Encode(frameBase64Buffer, frameBuffer.Bytes())

	const chunkSize = 4096
	for i := 0; i < len(frameBase64Buffer); i += chunkSize {
		end := i + chunkSize
		if end > len(frameBase64Buffer) {
			end = len(frameBase64Buffer)
		}
		m := 0
		if end < len(frameBase64Buffer) {
			m = 1
		}
		if i == 0 {
			fmt.Printf("\033_Gf=100,t=d,q=2,a=T,i=%d,m=%d;", id, m)
		} else {
			fmt.Printf("\033_Gm=%d;", m)
		}
		_, _ = os.Stdout.Write(frameBase64Buffer[i:end])
		fmt.Print("\033\\")
	}
}

func cleanupTerminal() {
	fmt.Print("\033_Ga=d,d=A,q=2\033\\")
	fmt.Print("\033[?25h")
	fmt.Print("\033[2J\033[H")
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

func dot(a, b vec3) float64 { return a.x*b.x + a.y*b.y + a.z*b.z }
func add(a, b vec3) vec3    { return vec3{a.x + b.x, a.y + b.y, a.z + b.z} }
func sub(a, b vec3) vec3    { return vec3{a.x - b.x, a.y - b.y, a.z - b.z} }
func scale(a vec3, s float64) vec3 {
	return vec3{a.x * s, a.y * s, a.z * s}
}
func cross(a, b vec3) vec3 {
	return vec3{
		x: a.y*b.z - a.z*b.y,
		y: a.z*b.x - a.x*b.z,
		z: a.x*b.y - a.y*b.x,
	}
}
func norm(a vec3) vec3 {
	l := math.Sqrt(a.x*a.x + a.y*a.y + a.z*a.z)
	if l <= 1e-12 {
		return vec3{}
	}
	inv := 1.0 / l
	return vec3{a.x * inv, a.y * inv, a.z * inv}
}

func length3(a vec3) float64 {
	return math.Sqrt(a.x*a.x + a.y*a.y + a.z*a.z)
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func clampf(v, lo, hi float64) float64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

func fitCameraDistanceForUnitCube(fovY, aspect, margin float64) float64 {
	if margin < 1 {
		margin = 1
	}
	if aspect <= 0 {
		aspect = 1
	}
	// Unit cube [-0.5, 0.5]^3 enclosing-sphere radius.
	sphereRadius := math.Sqrt(3.0) * 0.5

	halfY := fovY * 0.5
	halfX := math.Atan(math.Tan(halfY) * aspect)
	limitingHalfFOV := halfY
	if halfX < limitingHalfFOV {
		limitingHalfFOV = halfX
	}
	if limitingHalfFOV < 0.05 {
		limitingHalfFOV = 0.05
	}

	// Fit enclosing sphere into the narrowest frustum half-angle.
	d := sphereRadius / math.Sin(limitingHalfFOV)
	return d * margin
}

func wrapAngle(a float64) float64 {
	for a <= -math.Pi {
		a += 2 * math.Pi
	}
	for a > math.Pi {
		a -= 2 * math.Pi
	}
	return a
}

func shortestAngleDelta(from, to float64) float64 {
	return wrapAngle(to - from)
}

func newGPUVolumeRenderer(vol *volume) (*gpuVolumeRenderer, error) {
	r := &gpuVolumeRenderer{
		volNx: vol.nx,
		volNy: vol.ny,
		volNz: vol.nz,
	}
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
		Label: "gbrain-volume-compute.wgsl",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{
			Code: gbrainVolumeComputeWGSL,
		},
	})
	if err != nil {
		r.Close()
		return nil, err
	}
	defer module.Release()

	pipeline, err := r.device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label: "gbrain-volume-compute-pipeline",
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
		Label: "gbrain-params",
		Size:  80,
		Usage: wgpu.BufferUsage_Uniform | wgpu.BufferUsage_CopyDst,
	})
	if err != nil {
		r.Close()
		return nil, err
	}
	r.paramsBuffer = paramsBuffer

	voxelCount := len(vol.data)
	volumeRaw := make([]byte, voxelCount*4)
	for i, v := range vol.data {
		binary.LittleEndian.PutUint32(volumeRaw[i*4:], uint32(v))
	}

	volumeBuffer, err := r.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "gbrain-volume",
		Size:  uint64(len(volumeRaw)),
		Usage: wgpu.BufferUsage_Storage | wgpu.BufferUsage_CopyDst,
	})
	if err != nil {
		r.Close()
		return nil, err
	}
	r.volumeBuffer = volumeBuffer
	if err := r.queue.WriteBuffer(r.volumeBuffer, 0, volumeRaw); err != nil {
		r.Close()
		return nil, err
	}

	return r, nil
}

func (r *gpuVolumeRenderer) ensureBuffers(width, height int) error {
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
		Label: "gbrain-output",
		Size:  bufferSize,
		Usage: wgpu.BufferUsage_Storage | wgpu.BufferUsage_CopySrc,
	})
	if err != nil {
		return err
	}
	r.outputBuffer = outputBuffer

	readbackBuffer, err := r.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "gbrain-readback",
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
			{Binding: 0, Buffer: r.paramsBuffer, Size: 80},
			{Binding: 1, Buffer: r.volumeBuffer, Size: wgpu.WholeSize},
			{Binding: 2, Buffer: r.outputBuffer, Size: wgpu.WholeSize},
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

func (r *gpuVolumeRenderer) Render(width, height int, p gpuFrameParams) (*image.RGBA, error) {
	if err := r.ensureBuffers(width, height); err != nil {
		return nil, err
	}

	paramsRaw := make([]byte, 80)
	binary.LittleEndian.PutUint32(paramsRaw[0:], uint32(width))
	binary.LittleEndian.PutUint32(paramsRaw[4:], uint32(height))
	binary.LittleEndian.PutUint32(paramsRaw[8:], p.Samples)
	binary.LittleEndian.PutUint32(paramsRaw[12:], p.PaletteDensity)
	binary.LittleEndian.PutUint32(paramsRaw[16:], uint32(r.volNx))
	binary.LittleEndian.PutUint32(paramsRaw[20:], uint32(r.volNy))
	binary.LittleEndian.PutUint32(paramsRaw[24:], uint32(r.volNz))
	binary.LittleEndian.PutUint32(paramsRaw[28:], p.PaletteEdge)
	binary.LittleEndian.PutUint32(paramsRaw[32:], math.Float32bits(p.Opacity))
	binary.LittleEndian.PutUint32(paramsRaw[36:], math.Float32bits(p.Iso))
	binary.LittleEndian.PutUint32(paramsRaw[40:], math.Float32bits(p.EdgeBoost))
	binary.LittleEndian.PutUint32(paramsRaw[44:], math.Float32bits(float32(34.0*math.Pi/180.0)))
	binary.LittleEndian.PutUint32(paramsRaw[48:], math.Float32bits(p.Yaw))
	binary.LittleEndian.PutUint32(paramsRaw[52:], math.Float32bits(p.Pitch))
	binary.LittleEndian.PutUint32(paramsRaw[56:], math.Float32bits(p.Roll))
	binary.LittleEndian.PutUint32(paramsRaw[60:], math.Float32bits(float32(float64(width)/float64(height))))
	binary.LittleEndian.PutUint32(paramsRaw[64:], math.Float32bits(p.FitMargin))
	binary.LittleEndian.PutUint32(paramsRaw[68:], p.ColorMode)
	binary.LittleEndian.PutUint32(paramsRaw[72:], 0)
	binary.LittleEndian.PutUint32(paramsRaw[76:], 0)

	if err := r.queue.WriteBuffer(r.paramsBuffer, 0, paramsRaw); err != nil {
		return nil, err
	}

	encoder, err := r.device.CreateCommandEncoder(&wgpu.CommandEncoderDescriptor{Label: "gbrain-compute-encoder"})
	if err != nil {
		return nil, err
	}
	defer encoder.Release()

	pass := encoder.BeginComputePass(&wgpu.ComputePassDescriptor{Label: "gbrain-compute-pass"})
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

func (r *gpuVolumeRenderer) Close() {
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
	if r.volumeBuffer != nil {
		r.volumeBuffer.Release()
		r.volumeBuffer = nil
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

const gbrainVolumeComputeWGSL = `
struct Params {
    width: u32,
    height: u32,
    samples: u32,
	palette_density: u32,
    nx: u32,
    ny: u32,
    nz: u32,
	palette_edge: u32,
    opacity: f32,
    iso: f32,
    edge_boost: f32,
    fov_y: f32,
    yaw: f32,
    pitch: f32,
    roll: f32,
    aspect: f32,
    fit_margin: f32,
	color_mode: u32,
	_pad0: u32,
	_pad1: u32,
};

@group(0) @binding(0)
var<uniform> params: Params;

@group(0) @binding(1)
var<storage, read> volume: array<u32>;

@group(0) @binding(2)
var<storage, read_write> out_pixels: array<u32>;

fn clamp01(v: f32) -> f32 {
    return clamp(v, 0.0, 1.0);
}

fn pack_rgba(c: vec3<f32>) -> u32 {
    let r = u32(round(clamp01(c.x) * 255.0));
    let g = u32(round(clamp01(c.y) * 255.0));
    let b = u32(round(clamp01(c.z) * 255.0));
    return r | (g << 8u) | (b << 16u) | (255u << 24u);
}

fn safe_norm(v: vec3<f32>) -> vec3<f32> {
    let l = length(v);
    if (l <= 1e-6) {
        return vec3<f32>(0.0);
    }
    return v / l;
}

fn voxel_at(ix: u32, iy: u32, iz: u32) -> f32 {
    if (ix >= params.nx || iy >= params.ny || iz >= params.nz) {
        return 0.0;
    }
    let idx = (iz * params.ny + iy) * params.nx + ix;
    return f32(volume[idx]) / 255.0;
}

fn sample_volume_trilinear(u: f32, v: f32, w: f32) -> f32 {
    if (u < 0.0 || u > 1.0 || v < 0.0 || v > 1.0 || w < 0.0 || w > 1.0) {
        return 0.0;
    }
    let x = u * f32(params.nx - 1u);
    let y = v * f32(params.ny - 1u);
    let z = w * f32(params.nz - 1u);

    let x0 = u32(floor(x));
    let y0 = u32(floor(y));
    let z0 = u32(floor(z));
    let x1 = min(x0 + 1u, params.nx - 1u);
    let y1 = min(y0 + 1u, params.ny - 1u);
    let z1 = min(z0 + 1u, params.nz - 1u);

    let tx = x - f32(x0);
    let ty = y - f32(y0);
    let tz = z - f32(z0);

    let c000 = voxel_at(x0, y0, z0);
    let c100 = voxel_at(x1, y0, z0);
    let c010 = voxel_at(x0, y1, z0);
    let c110 = voxel_at(x1, y1, z0);
    let c001 = voxel_at(x0, y0, z1);
    let c101 = voxel_at(x1, y0, z1);
    let c011 = voxel_at(x0, y1, z1);
    let c111 = voxel_at(x1, y1, z1);

    let c00 = mix(c000, c100, tx);
    let c10 = mix(c010, c110, tx);
    let c01 = mix(c001, c101, tx);
    let c11 = mix(c011, c111, tx);
    let c0 = mix(c00, c10, ty);
    let c1 = mix(c01, c11, ty);
    return mix(c0, c1, tz);
}

fn sample_gradient(u: f32, v: f32, w: f32, h: f32) -> vec3<f32> {
    let hh = max(h, 1e-5);
    let x0 = sample_volume_trilinear(u - hh, v, w);
    let x1 = sample_volume_trilinear(u + hh, v, w);
    let y0 = sample_volume_trilinear(u, v - hh, w);
    let y1 = sample_volume_trilinear(u, v + hh, w);
    let z0 = sample_volume_trilinear(u, v, w - hh);
    let z1 = sample_volume_trilinear(u, v, w + hh);
    let inv = 0.5 / hh;
    return vec3<f32>((x1 - x0) * inv, (y1 - y0) * inv, (z1 - z0) * inv);
}

fn gradient3(t: f32, c0: vec3<f32>, c1: vec3<f32>, c2: vec3<f32>) -> vec3<f32> {
    if (t < 0.5) {
        return mix(c0, c1, t * 2.0);
    }
    return mix(c1, c2, (t - 0.5) * 2.0);
}

fn color_from_palette(t_in: f32, palette: u32) -> vec3<f32> {
    let t = clamp01(t_in);
    switch palette {
        case 1u {
            return gradient3(t,
                vec3<f32>(0.02, 0.01, 0.04),
                vec3<f32>(0.75, 0.12, 0.03),
                vec3<f32>(1.00, 0.88, 0.18));
        }
        case 2u {
            return gradient3(t,
                vec3<f32>(0.01, 0.06, 0.10),
                vec3<f32>(0.08, 0.44, 0.70),
                vec3<f32>(0.80, 0.97, 1.00));
        }
        case 3u {
            return gradient3(t,
                vec3<f32>(0.01, 0.05, 0.02),
                vec3<f32>(0.08, 0.45, 0.16),
                vec3<f32>(0.86, 0.97, 0.78));
        }
        case 4u {
            let g = 0.08 + 0.92 * t;
            return vec3<f32>(g, g, g);
        }
        default {
            return gradient3(t,
                vec3<f32>(0.06, 0.03, 0.14),
                vec3<f32>(0.42, 0.22, 0.70),
                vec3<f32>(0.97, 0.80, 0.98));
        }
    }
}

fn color_from_mode(mode: u32, palette_density: u32, palette_edge: u32, local: f32, edge: f32, depth: f32, n: vec3<f32>, alpha: f32) -> vec3<f32> {
	switch mode {
		case 1u {
			return color_from_palette(edge, palette_edge);
		}
		case 2u {
			let dcol = color_from_palette(local, palette_density);
			let ecol = color_from_palette(edge, palette_edge);
			let mix_e = clamp(0.42 * edge + 0.18, 0.0, 0.78);
			return mix(dcol, ecol, mix_e);
		}
		case 3u {
			return color_from_palette(1.0 - clamp01(depth), palette_density);
		}
		case 4u {
			return clamp(n * 0.5 + vec3<f32>(0.5), vec3<f32>(0.0), vec3<f32>(1.0));
		}
		case 5u {
			return color_from_palette(clamp01(alpha / 0.9), palette_density);
		}
		default {
			return color_from_palette(local, palette_density);
		}
	}
}

fn intersect_unit_box(origin: vec3<f32>, dir: vec3<f32>) -> vec2<f32> {
    let box_min = vec3<f32>(-0.5, -0.5, -0.5);
    let box_max = vec3<f32>(0.5, 0.5, 0.5);
    var t0 = -1e30;
    var t1 = 1e30;

    if (abs(dir.x) < 1e-6) {
        if (origin.x < box_min.x || origin.x > box_max.x) {
            return vec2<f32>(1.0, -1.0);
        }
    } else {
        var a = (box_min.x - origin.x) / dir.x;
        var b = (box_max.x - origin.x) / dir.x;
        if (a > b) { let tmp = a; a = b; b = tmp; }
        t0 = max(t0, a);
        t1 = min(t1, b);
    }

    if (abs(dir.y) < 1e-6) {
        if (origin.y < box_min.y || origin.y > box_max.y) {
            return vec2<f32>(1.0, -1.0);
        }
    } else {
        var a = (box_min.y - origin.y) / dir.y;
        var b = (box_max.y - origin.y) / dir.y;
        if (a > b) { let tmp = a; a = b; b = tmp; }
        t0 = max(t0, a);
        t1 = min(t1, b);
    }

    if (abs(dir.z) < 1e-6) {
        if (origin.z < box_min.z || origin.z > box_max.z) {
            return vec2<f32>(1.0, -1.0);
        }
    } else {
        var a = (box_min.z - origin.z) / dir.z;
        var b = (box_max.z - origin.z) / dir.z;
        if (a > b) { let tmp = a; a = b; b = tmp; }
        t0 = max(t0, a);
        t1 = min(t1, b);
    }

    return vec2<f32>(t0, t1);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    let idx = gid.y * params.width + gid.x;
	let bg = vec3<f32>(3.0 / 255.0, 7.0 / 255.0, 12.0 / 255.0);

    let half_y = params.fov_y * 0.5;
    let half_x = atan(tan(half_y) * params.aspect);
    let limiting = max(0.05, min(half_y, half_x));
    let sphere_r = sqrt(3.0) * 0.5;
    let radius = (sphere_r / sin(limiting)) * params.fit_margin;

    let cam = vec3<f32>(
        radius * cos(params.pitch) * cos(params.yaw),
        radius * sin(params.pitch),
        radius * cos(params.pitch) * sin(params.yaw));
    let forward = safe_norm(-cam);
    var up_world = vec3<f32>(0.0, 1.0, 0.0);
    if (abs(dot(forward, up_world)) > 0.98) {
        up_world = vec3<f32>(0.0, 0.0, 1.0);
    }
    var right = safe_norm(cross(forward, up_world));
    var up = safe_norm(cross(right, forward));
    if (abs(params.roll) > 1e-6) {
        let cr = cos(params.roll);
        let sr = sin(params.roll);
        let r2 = right * cr + up * sr;
        let u2 = up * cr - right * sr;
        right = r2;
        up = u2;
    }

    let tan_half = tan(half_y);
    let sx = (((f32(gid.x) + 0.5) / f32(params.width)) * 2.0 - 1.0) * params.aspect * tan_half;
    let sy = (1.0 - ((f32(gid.y) + 0.5) / f32(params.height)) * 2.0) * tan_half;
    let dir = safe_norm(forward + right * sx + up * sy);

    let hit = intersect_unit_box(cam, dir);
    var t_near = hit.x;
    let t_far = hit.y;
    if (t_far < t_near) {
        out_pixels[idx] = pack_rgba(bg);
        return;
    }
    if (t_near < 0.0) {
        t_near = 0.0;
    }

    let step_size = sqrt(3.0) / f32(max(1u, params.samples));
    let grad_step = 1.35 / f32(max(params.nx, max(params.ny, params.nz)));
    let light_dir = safe_norm(forward + right * -0.20 + up * 0.26);

    var acc = vec3<f32>(0.0);
    var acc_a = 0.0;
    var prev = 0.0;
    var t = t_near;
	let range_len = max(t_far - t_near, 1e-6);
    loop {
        if (t >= t_far || acc_a > 0.965) {
            break;
        }

        let p = cam + dir * t;
        let u = p.x + 0.5;
        let v = p.y + 0.5;
        let w = p.z + 0.5;
        let d = sample_volume_trilinear(u, v, w);
        if (d <= params.iso * 0.45) {
            prev = d;
            t = t + step_size;
            continue;
        }

        let grad = sample_gradient(u, v, w, grad_step);
        let edge = clamp01(length(grad) * params.edge_boost);
        let delta = max(d - prev, 0.0);
        prev = d;

        let local = clamp01((d - params.iso) / max(1e-4, 1.0 - params.iso));
        let n = safe_norm(grad);
        let shade = 0.32 + 0.68 * max(dot(n, light_dir), 0.0);

        var a = params.opacity * 0.66 * (0.10 + 0.90 * local);
        a = a * (0.42 + edge * 0.92);
        a = a * (1.0 + delta * 1.45);
        a = min(a, 0.9);

		let depth = clamp01((t - t_near) / range_len);
		let base_col = color_from_mode(params.color_mode, params.palette_density, params.palette_edge, local, edge, depth, n, a);
        let colour_boost = 0.78 + 0.38 * edge;
		var shade_factor = shade;
		if (params.color_mode == 4u) {
			shade_factor = 1.0;
		}
		let col = clamp(base_col * shade_factor * colour_boost, vec3<f32>(0.0), vec3<f32>(1.0));

        let one_minus = 1.0 - acc_a;
        acc = acc + one_minus * a * col;
        acc_a = acc_a + one_minus * a;

        t = t + step_size;
    }

    let out_col = acc + (1.0 - acc_a) * bg;
    out_pixels[idx] = pack_rgba(out_col);
}
`
