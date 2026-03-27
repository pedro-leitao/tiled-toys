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

type vec3 struct{ x, y, z float64 }

type particle struct {
	pos   vec3
	trail []vec3
	next  int
	full  bool
	hue   float64
}

type paletteType int

const (
	paletteTwilight paletteType = iota
	paletteFire
	paletteIce
	paletteForest
	paletteMono
)

var framePNGEncoder = png.Encoder{CompressionLevel: png.BestSpeed}
var frameBuffer bytes.Buffer
var frameBase64Buffer []byte

func main() {
	spm := flag.Int("spm", 900, "Simulation steps per minute")
	systemName := flag.String("system", "lorenz", "Attractor system: lorenz|rossler")
	cloudSize := flag.Int("cloud", 48, "Number of particles")
	trailLen := flag.Int("trail", 180, "Trail length per particle")
	dt := flag.Float64("dt", 0.005, "Integrator step size")
	substeps := flag.Int("substeps", 4, "Integration updates per step")
	rotationSpeed := flag.Float64("rotation-speed", 0.28, "Camera rotation speed")
	paletteName := flag.String("palette", "twilight", "Color palette: twilight|fire|ice|forest|mono")
	frameStride := flag.Int("frame-stride", 1, "Render one frame every N simulation steps")
	flag.Parse()

	if *spm < 1 {
		*spm = 1
	}
	if *cloudSize < 1 {
		*cloudSize = 1
	}
	if *cloudSize > 256 {
		*cloudSize = 256
	}
	if *trailLen < 8 {
		*trailLen = 8
	}
	if *trailLen > 600 {
		*trailLen = 600
	}
	if *dt <= 0 {
		*dt = 0.005
	}
	if *substeps < 1 {
		*substeps = 1
	}
	if *substeps > 20 {
		*substeps = 20
	}
	if *frameStride < 1 {
		*frameStride = 1
	}

	rnd := rand.New(rand.NewSource(time.Now().UnixNano()))
	system := parseSystem(*systemName)
	palette := parsePalette(*paletteName)
	parts := make([]particle, *cloudSize)
	for i := range parts {
		parts[i].trail = make([]vec3, *trailLen)
		parts[i].hue = 0.52 + 0.35*float64(i)/float64(len(parts))
		if system == "rossler" {
			parts[i].pos = vec3{0.2 + (rnd.Float64()-0.5)*0.4, (rnd.Float64() - 0.5) * 0.4, (rnd.Float64() - 0.5) * 0.4}
		} else {
			parts[i].pos = vec3{0.1 + (rnd.Float64()-0.5)*0.3, (rnd.Float64() - 0.5) * 0.3, 20 + (rnd.Float64()-0.5)*0.3}
		}
		for t := 0; t < *trailLen; t++ {
			stepParticle(&parts[i], system, *dt)
			addTrail(&parts[i])
		}
	}

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

	frame := int64(0)
	step := 0
	currentID := 1
	previousID := 2
	camOffsetX := 0.0
	camOffsetY := 0.0
	camScale := 1.0
	camReady := false

	render := func() {
		w, h := getTermPixels()
		if w < 16 || h < 16 {
			return
		}

		img := image.NewRGBA(image.Rect(0, 0, w, h))
		fillBackground(img, color.RGBA{R: 3, G: 6, B: 10, A: 255})

		yaw := float64(frame) * *rotationSpeed / 60.0
		pitch := 0.5 + 0.12*math.Sin(float64(frame)*0.01)

		count := 0
		sumX := 0.0
		sumY := 0.0
		for i := range parts {
			n := trailCount(&parts[i])
			for j := 0; j < n; j++ {
				pt := trailAt(&parts[i], j)
				sx, sy, _ := projectPoint(pt, yaw, pitch)
				sumX += sx
				sumY += sy
				count++
			}
		}
		if count == 0 {
			return
		}

		targetOffsetX := sumX / float64(count)
		targetOffsetY := sumY / float64(count)
		maxDX := 0.001
		maxDY := 0.001
		for i := range parts {
			n := trailCount(&parts[i])
			for j := 0; j < n; j++ {
				pt := trailAt(&parts[i], j)
				sx, sy, _ := projectPoint(pt, yaw, pitch)
				dx := math.Abs(sx - targetOffsetX)
				dy := math.Abs(sy - targetOffsetY)
				if dx > maxDX {
					maxDX = dx
				}
				if dy > maxDY {
					maxDY = dy
				}
			}
		}

		targetScaleX := (float64(w) * 0.44) / maxDX
		targetScaleY := (float64(h) * 0.40) / maxDY
		targetScale := math.Min(targetScaleX, targetScaleY)
		if targetScale < 2 {
			targetScale = 2
		}
		maxScale := float64(minInt(w, h)) * 4.0
		if targetScale > maxScale {
			targetScale = maxScale
		}

		if !camReady {
			camOffsetX = targetOffsetX
			camOffsetY = targetOffsetY
			camScale = targetScale
			camReady = true
		} else {
			camOffsetX += (targetOffsetX - camOffsetX) * 0.10
			camOffsetY += (targetOffsetY - camOffsetY) * 0.10
			camScale += (targetScale - camScale) * 0.12
		}

		for i := range parts {
			n := trailCount(&parts[i])
			for j := 0; j < n; j++ {
				pt := trailAt(&parts[i], j)
				sx, sy, depth := projectPoint(pt, yaw, pitch)
				x := int(float64(w)/2 + (sx-camOffsetX)*camScale)
				y := int(float64(h)/2 - (sy-camOffsetY)*camScale)
				if x < 0 || y < 0 || x >= w || y >= h {
					continue
				}

				t := float64(j+1) / float64(n)
				fade := clamp01(0.08 + 0.92*t*t*depth)
				col := colorFromPalette(palette, parts[i].hue, t, fade)
				radius := 1
				if j == n-1 {
					radius = 2
					col = color.RGBA{R: 235, G: 245, B: 255, A: 255}
				}
				drawDot(img, x, y, radius, col)
			}
		}

		fmt.Print("\033[H")
		printKittyImage(img, currentID)
		fmt.Printf("\033_Ga=d,d=i,q=2,i=%d\033\\", previousID)
		currentID, previousID = previousID, currentID
	}

	render()
	for {
		select {
		case <-ticker.C:
			for s := 0; s < *substeps; s++ {
				for i := range parts {
					stepParticle(&parts[i], system, *dt)
					addTrail(&parts[i])
				}
			}
			frame++
			step++
			if step%*frameStride == 0 {
				render()
			}
		case <-resizeTicker.C:
			render()
		}
	}
}

func parseSystem(s string) string {
	s = strings.TrimSpace(strings.ToLower(s))
	if s == "rossler" || s == "rössler" {
		return "rossler"
	}
	return "lorenz"
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

func stepParticle(p *particle, system string, dt float64) {
	x, y, z := p.pos.x, p.pos.y, p.pos.z
	if system == "rossler" {
		dx, dy, dz := -y-z, x+0.2*y, 0.2+z*(x-5.7)
		x, y, z = x+dx*dt, y+dy*dt, z+dz*dt
	} else {
		dx, dy, dz := 10*(y-x), x*(28-z)-y, x*y-(8.0/3.0)*z
		x, y, z = x+dx*dt, y+dy*dt, z+dz*dt
	}
	p.pos = vec3{x, y, z}
}

func addTrail(p *particle) { p.trail[p.next] = p.pos; p.next++; if p.next >= len(p.trail) { p.next = 0; p.full = true } }
func trailCount(p *particle) int { if p.full { return len(p.trail) }; return p.next }
func trailAt(p *particle, i int) vec3 { if !p.full { return p.trail[i] }; idx := p.next + i; if idx >= len(p.trail) { idx -= len(p.trail) }; return p.trail[idx] }

func projectPoint(p vec3, yaw, pitch float64) (float64, float64, float64) {
	cy, sy := math.Cos(yaw), math.Sin(yaw)
	cp, sp := math.Cos(pitch), math.Sin(pitch)
	x1 := p.x*cy - p.y*sy
	y1 := p.x*sy + p.y*cy
	y2 := y1*cp - p.z*sp
	z2 := y1*sp + p.z*cp
	den := 130.0 - z2
	if den < 5 {
		den = 5
	}
	k := 130.0 / den
	depth := 0.55 + 0.45*k
	if depth < 0.25 {
		depth = 0.25
	}
	if depth > 1.5 {
		depth = 1.5
	}
	return x1 * k, y2 * k, depth
}

func fillBackground(img *image.RGBA, c color.RGBA) { b := img.Bounds(); for y := b.Min.Y; y < b.Max.Y; y++ { for x := b.Min.X; x < b.Max.X; x++ { img.SetRGBA(x, y, c) } } }
func drawDot(img *image.RGBA, x, y, r int, c color.RGBA) { b := img.Bounds(); for dy := -r; dy <= r; dy++ { for dx := -r; dx <= r; dx++ { if dx*dx+dy*dy > r*r { continue }; px, py := x+dx, y+dy; if px < b.Min.X || py < b.Min.Y || px >= b.Max.X || py >= b.Max.Y { continue }; o := img.PixOffset(px, py); a := float64(c.A) / 255.0; ia := 1 - a; img.Pix[o+0] = uint8(float64(c.R)*a + float64(img.Pix[o+0])*ia + 0.5); img.Pix[o+1] = uint8(float64(c.G)*a + float64(img.Pix[o+1])*ia + 0.5); img.Pix[o+2] = uint8(float64(c.B)*a + float64(img.Pix[o+2])*ia + 0.5); img.Pix[o+3] = 255 } } }
func hsvToRGB(h, s, v float64) color.RGBA { h = math.Mod(h, 1); if h < 0 { h += 1 }; s, v = clamp01(s), clamp01(v); hh := h * 6; i := int(hh); f := hh - float64(i); p, q, t := v*(1-s), v*(1-s*f), v*(1-s*(1-f)); var r, g, b float64; switch i % 6 { case 0: r, g, b = v, t, p; case 1: r, g, b = q, v, p; case 2: r, g, b = p, v, t; case 3: r, g, b = p, q, v; case 4: r, g, b = t, p, v; default: r, g, b = v, p, q }; return color.RGBA{R: uint8(r*255 + 0.5), G: uint8(g*255 + 0.5), B: uint8(b*255 + 0.5), A: 220} }
func colorFromPalette(p paletteType, hue, trailT, fade float64) color.RGBA {
	t := clamp01(trailT)
	pulse := 0.5 + 0.5*math.Sin(6.28318*(t*1.7+0.1))
	spark := 0.5 + 0.5*math.Sin(6.28318*(t*4.4+0.8))

	var r, g, b float64
	switch p {
	case paletteFire:
		r = 0.10 + 0.95*t + 0.20*spark
		g = 0.02 + 0.45*t + 0.25*pulse
		b = 0.01 + 0.12*t
	case paletteIce:
		r = 0.04 + 0.25*t + 0.10*spark
		g = 0.14 + 0.55*t + 0.20*pulse
		b = 0.22 + 0.95*t
	case paletteForest:
		r = 0.03 + 0.20*t + 0.10*pulse
		g = 0.10 + 0.75*t + 0.25*spark
		b = 0.03 + 0.28*t
	case paletteMono:
		gMono := clamp01(0.06 + 0.92*t + 0.06*pulse)
		v := clamp01(gMono * fade)
		return color.RGBA{R: uint8(v*255 + 0.5), G: uint8(v*255 + 0.5), B: uint8(v*255 + 0.5), A: 220}
	default:
		// twilight with subtle per-particle hue variation
		base := hsvToRGB(math.Mod(hue+0.08*t, 1.0), 0.85, clamp01(0.35+0.65*fade))
		r = float64(base.R) / 255.0
		g = float64(base.G) / 255.0
		b = float64(base.B) / 255.0
	}

	r = clamp01(r * fade)
	g = clamp01(g * fade)
	b = clamp01(b * fade)
	return color.RGBA{R: uint8(r*255 + 0.5), G: uint8(g*255 + 0.5), B: uint8(b*255 + 0.5), A: 220}
}
func clamp01(v float64) float64 { if v < 0 { return 0 }; if v > 1 { return 1 }; return v }
func minInt(a, b int) int { if a < b { return a }; return b }

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

func cleanupTerminal() { fmt.Print("\033_Ga=d,d=A,q=2\033\\\033[?25h\033[2J\033[H") }

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