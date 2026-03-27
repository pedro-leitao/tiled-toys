package main

import (
	"bytes"
	"encoding/base64"
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

	rnd *rand.Rand
}

type paletteType int
type fractalMathMode int

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
	fixedShift   = 28
	fixedOne     = int64(1 << fixedShift)
	fixedEscape2 = int64(16) << fixedShift
)

var framePNGEncoder = png.Encoder{CompressionLevel: png.BestSpeed}
var frameBuffer bytes.Buffer
var frameBase64Buffer []byte

var focusPoints = []FocusPoint{
	// Full Overviews (Centered to fit the whole set nicely)
	{X: -0.65, Y: 0.0, Span: 2.1},     // The standard full view
	{X: -1.0, Y: 0.0, Span: 1.2},      // Period-2 bulb and main antenna

	// Seahorse Valley (The cleft between the main cardioid and period-2 bulb)
	// Highly reliable, incredibly rich in spirals
	{X: -0.745, Y: 0.1, Span: 0.05},       // Valley entrance
	{X: -0.748, Y: 0.1, Span: 0.005},      // Medium zoom into the seahorses
	{X: -0.7488, Y: 0.093, Span: 0.0005},  // Deep spiral (Safe for float32)
	{X: -0.74364, Y: 0.13182, Span: 0.00005}, // Extreme micro-detail boundary

	// Elephant Valley (The right-side cleft of the main cardioid)
	// Features branching, tree-like structures rather than spirals
	{X: 0.275, Y: 0.005, Span: 0.05},      // Valley overview
	{X: 0.274, Y: 0.006, Span: 0.005},     // Mid-level elephant trunks
	{X: 0.282, Y: 0.010, Span: 0.002},     // Deep trunk detail

	// Scepter Valley / Period-3 Bulb Boundary
	{X: -1.25066, Y: 0.02012, Span: 0.01},   // Nice chaotic boundary
	{X: -1.310, Y: 0.030, Span: 0.02},       // Period-3 bulb edge
	{X: -1.360, Y: 0.005, Span: 0.05},       // Scepter valley overview

	// Triple-Spiral Valley / Top edge
	{X: -0.10, Y: 0.90, Span: 0.2},          // Approach to the top boundary
	{X: -0.1011, Y: 0.9563, Span: 0.02},     // Top spiraling chaotic zone
	{X: -0.156520, Y: 1.03225, Span: 0.005}, // High-contrast Misiurewicz point
	
	// Needle / Filament Detail (The long antenna on the left)
	{X: -1.770, Y: 0.0, Span: 0.05},         // Main antenna context
	{X: -1.768778, Y: 0.001738, Span: 0.005}, // Island on the antenna
	{X: -1.94, Y: 0.0, Span: 0.1},           // Extreme left tip context
	{X: -1.999985, Y: 0.0, Span: 0.00005},   // Deep left tip mini-brot

	// Guaranteed Minibrots (Perfectly centered self-similar copies)
	{X: -1.7687788, Y: 0.0, Span: 0.01},     // Largest minibrot on the needle
	{X: -0.1528, Y: 1.0397, Span: 0.05},     // Top edge minibrot
	{X: -0.79319, Y: 0.16093, Span: 0.005},  // Seahorse valley minibrot
	{X: 0.3533, Y: 0.0988, Span: 0.005},     // Elephant valley minibrot
}

func main() {
	spm := flag.Int("spm", 360, "Steps per minute")
	paletteName := flag.String("palette", "twilight", "Color palette: twilight|fire|ice|forest|mono")
	mathModeName := flag.String("math-mode", "fixed", "Mandelbrot kernel: fixed|float")
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
		rnd:       rand.New(rand.NewSource(time.Now().UnixNano())),
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
	v.area = focusPoints[v.rnd.Intn(len(focusPoints))]
	v.centerX = v.area.X
	v.centerY = v.area.Y
	v.spanX = v.area.Span
	v.refineStep = 0
	v.holdStep = 0
}

func (v *Viewer) renderFrame(w, h int) image.Image {
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	progress := float32(v.refineStep) / float32(v.refineMax)
	if progress < 0 {
		progress = 0
	}
	if progress > 1 {
		progress = 1
	}
	block := blockSizeFromProgress(progress, v.maxBlock)
	invW := 1.0 / float32(w)
	invH := 1.0 / float32(h)
	spanY := v.spanX * (float32(h) * invW)
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
		y := v.centerY + ((float32(sy)*invH)-0.5)*spanY

		for bx := 0; bx < w; bx += block {
			xEnd := bx + block
			if xEnd > w {
				xEnd = w
			}

			sx := bx + (xEnd-bx)/2
			x := v.centerX + ((float32(sx)*invW)-0.5)*v.spanX
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
