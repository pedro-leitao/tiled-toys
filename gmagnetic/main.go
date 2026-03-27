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

type vec2 struct{ x, y float64 }

type particle struct {
	pos    vec2
	vel    vec2
	trail  []vec2
	next   int
	full   bool
	charge float64
}

type paletteType int

const (
	paletteAurora paletteType = iota
	paletteFire
	paletteIce
	paletteMono
)

var framePNGEncoder = png.Encoder{CompressionLevel: png.BestSpeed}
var frameBuffer bytes.Buffer
var frameBase64Buffer []byte

func main() {
	spm := flag.Int("spm", 900, "Simulation steps per minute")
	particleCount := flag.Int("particles", 420, "Number of particles")
	trailLen := flag.Int("trail", 80, "Trail length per particle")
	dt := flag.Float64("dt", 0.016, "Integrator step size")
	substeps := flag.Int("substeps", 2, "Integration updates per simulation step")
	fieldStrength := flag.Float64("field-strength", 1.0, "Global magnetic field strength multiplier")
	rotationSpeed := flag.Float64("rotation-speed", 0.28, "Magnetic field rotation speed")
	convergeSpeed := flag.Float64("converge-speed", 0.08, "Pole convergence speed")
	paletteName := flag.String("palette", "aurora", "Color palette: aurora|fire|ice|mono")
	frameStride := flag.Int("frame-stride", 1, "Render one frame every N simulation steps")
	flag.Parse()

	if *spm < 1 {
		*spm = 1
	}
	if *particleCount < 32 {
		*particleCount = 32
	}
	if *particleCount > 2400 {
		*particleCount = 2400
	}
	if *trailLen < 12 {
		*trailLen = 12
	}
	if *trailLen > 220 {
		*trailLen = 220
	}
	if *dt <= 0 {
		*dt = 0.016
	}
	if *substeps < 1 {
		*substeps = 1
	}
	if *substeps > 12 {
		*substeps = 12
	}
	if *fieldStrength <= 0 {
		*fieldStrength = 1.0
	}
	if *convergeSpeed < 0 {
		*convergeSpeed = 0
	}
	if *frameStride < 1 {
		*frameStride = 1
	}

	rnd := rand.New(rand.NewSource(time.Now().UnixNano()))
	palette := parsePalette(*paletteName)
	parts := make([]particle, *particleCount)
	for i := range parts {
		parts[i].trail = make([]vec2, *trailLen)
		parts[i].charge = 1.0
		if rnd.Float64() < 0.5 {
			parts[i].charge = -1.0
		}
		parts[i].pos = vec2{x: rnd.Float64(), y: rnd.Float64()}
		angle := rnd.Float64() * math.Pi * 2
		speed := 0.05 + rnd.Float64()*0.22
		parts[i].vel = vec2{x: math.Cos(angle) * speed, y: math.Sin(angle) * speed}
		for t := 0; t < *trailLen; t++ {
			parts[i].trail[t] = parts[i].pos
		}
		parts[i].full = true
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

	step := 0
	frame := int64(0)
	currentID := 1
	previousID := 2

	render := func() {
		w, h := getTermPixels()
		if w < 16 || h < 16 {
			return
		}

		img := image.NewRGBA(image.Rect(0, 0, w, h))
		fillBackground(img, color.RGBA{R: 3, G: 8, B: 13, A: 255})
			drawFieldOverlay(img, frame, *fieldStrength, *rotationSpeed, *convergeSpeed)
		drawParticles(img, parts, palette, frame)

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
					stepParticle(&parts[i], *dt/float64(*substeps), frame, *fieldStrength, *rotationSpeed, *convergeSpeed, rnd)
					addTrail(&parts[i])
				}
				frame++
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

func parsePalette(s string) paletteType {
	s = strings.TrimSpace(strings.ToLower(s))
	switch s {
	case "fire":
		return paletteFire
	case "ice":
		return paletteIce
	case "mono", "monochrome":
		return paletteMono
	default:
		return paletteAurora
	}
}

func stepParticle(p *particle, dt float64, frame int64, strength, rotationSpeed, convergeSpeed float64, rnd *rand.Rand) {
	bx, by, bz := magneticField(p.pos.x, p.pos.y, frame, rotationSpeed, convergeSpeed)
	bMag := math.Sqrt(bx*bx + by*by)
	if bMag > 1.0 {
		s := 1.0 / bMag
		bx *= s
		by *= s
	}

	// Lorentz turn + gentle drift along local field-lines (tangent), not into poles.
	ax := p.charge*p.vel.y*bz - 0.08*by
	ay := -p.charge*p.vel.x*bz + 0.08*bx

	// Very weak centering to avoid long-term wall accumulation.
	ax += -(p.pos.x - 0.5) * 0.03
	ay += -(p.pos.y - 0.5) * 0.03

	ax *= strength
	ay *= strength

	ax += (rnd.Float64()-0.5)*0.010
	ay += (rnd.Float64()-0.5)*0.010

	const damping = 0.992
	p.vel.x = (p.vel.x+ax*dt)*damping
	p.vel.y = (p.vel.y+ay*dt)*damping

	speed2 := p.vel.x*p.vel.x + p.vel.y*p.vel.y
	maxSpeed := 0.70
	if speed2 > maxSpeed*maxSpeed {
		s := maxSpeed / math.Sqrt(speed2)
		p.vel.x *= s
		p.vel.y *= s
	}

	p.pos.x += p.vel.x * dt
	p.pos.y += p.vel.y * dt

	if p.pos.x < 0 {
		p.pos.x = 0
		p.vel.x *= -0.7
	}
	if p.pos.x > 1 {
		p.pos.x = 1
		p.vel.x *= -0.7
	}
	if p.pos.y < 0 {
		p.pos.y = 0
		p.vel.y *= -0.7
	}
	if p.pos.y > 1 {
		p.pos.y = 1
		p.vel.y *= -0.7
	}
}

func magneticField(nx, ny float64, frame int64, rotationSpeed, convergeSpeed float64) (float64, float64, float64) {
	cx := nx*2 - 1
	cy := ny*2 - 1
	theta := float64(frame) * rotationSpeed / 60.0
	ct := math.Cos(theta)
	st := math.Sin(theta)

	t := float64(frame) / 60.0
	startSep := 0.46
	minSep := 0.10
	sep := minSep + (startSep-minSep)*math.Exp(-convergeSpeed*t)
	sep *= 1.0 + 0.10*math.Sin(float64(frame)*0.006)
	if sep < minSep {
		sep = minSep
	}

	northX := -sep * ct
	northY := -sep * st
	southX := sep * ct
	southY := sep * st

	nxv := cx - northX
	nyv := cy - northY
	sxv := cx - southX
	syv := cy - southY

	rn2 := nxv*nxv + nyv*nyv + 0.003
	rs2 := sxv*sxv + syv*syv + 0.003

	rn := math.Sqrt(rn2)
	rs := math.Sqrt(rs2)

	invN := 1.0 / (rn2 * rn)
	invS := 1.0 / (rs2 * rs)

	bx := nxv*invN - sxv*invS
	by := nyv*invN - syv*invS

	axisProj := cx*ct + cy*st
	bz := 0.40*(1.0/rn2-1.0/rs2) + 0.06*math.Sin(4*axisProj+float64(frame)*0.013)

	return bx, by, bz
}

func drawFieldOverlay(img *image.RGBA, frame int64, strength, rotationSpeed, convergeSpeed float64) {
	w := img.Bounds().Dx()
	h := img.Bounds().Dy()
	if w < 8 || h < 8 {
		return
	}

	grid := 40
	if w < 500 || h < 360 {
		grid = 30
	}
	if w < 320 || h < 240 {
		grid = 24
	}

	for y := grid / 2; y < h; y += grid {
		for x := grid / 2; x < w; x += grid {
			nx := float64(x) / float64(w)
			ny := float64(y) / float64(h)
			bx, by, _ := magneticField(nx, ny, frame, rotationSpeed, convergeSpeed)
			mag := math.Sqrt(bx*bx + by*by)
			if mag < 1e-9 {
				continue
			}

			dx := bx / mag
			dy := by / mag
			seg := 6.0 + 8.0*math.Min(mag*1.2*strength, 1.0)

			x0 := int(float64(x) - dx*seg)
			y0 := int(float64(y) - dy*seg)
			x1 := int(float64(x) + dx*seg)
			y1 := int(float64(y) + dy*seg)

			alpha := uint8(36 + int(74*math.Min(mag*strength, 1.0)))
			drawLine(img, x0, y0, x1, y1, color.RGBA{R: 130, G: 130, B: 130, A: alpha})
		}
	}
}

func drawParticles(img *image.RGBA, parts []particle, palette paletteType, frame int64) {
	w := img.Bounds().Dx()
	h := img.Bounds().Dy()
	for i := range parts {
		n := trailCount(&parts[i])
		if n == 0 {
			continue
		}

		speed := math.Sqrt(parts[i].vel.x*parts[i].vel.x + parts[i].vel.y*parts[i].vel.y)
		for j := 0; j < n; j++ {
			pt := trailAt(&parts[i], j)
			x := int(pt.x * float64(w-1))
			y := int(pt.y * float64(h-1))
			if x < 0 || y < 0 || x >= w || y >= h {
				continue
			}

			t := float64(j+1) / float64(n)
			fade := 0.05 + 0.95*t*t
			col := particleColor(palette, parts[i].charge, t, fade, speed, frame, i)
			radius := 1
			if j == n-1 {
				radius = 2
				if parts[i].charge > 0 {
					col = color.RGBA{R: 255, G: 238, B: 200, A: 240}
				} else {
					col = color.RGBA{R: 190, G: 235, B: 255, A: 240}
				}
			}
			drawDot(img, x, y, radius, col)
		}
	}
}

func particleColor(p paletteType, charge, trailT, fade, speed float64, frame int64, seed int) color.RGBA {
	t := clamp01(trailT)
	pulse := 0.5 + 0.5*math.Sin(6.28318*(t*2.1+float64(seed)*0.013+float64(frame)*0.0008))
	kinetic := clamp01(speed / 0.70)
	if charge < 0 {
		pulse = 1 - pulse
	}

	var r, g, b float64
	switch p {
	case paletteFire:
		r = 0.18 + 0.78*t + 0.20*kinetic
		g = 0.06 + 0.45*t + 0.20*pulse
		b = 0.02 + 0.12*t
	case paletteIce:
		r = 0.06 + 0.24*t
		g = 0.22 + 0.58*t + 0.16*pulse
		b = 0.28 + 0.66*t + 0.18*kinetic
	case paletteMono:
		v := clamp01(0.08 + 0.88*t + 0.08*pulse)
		r, g, b = v, v, v
	default:
		if charge > 0 {
			r = 0.20 + 0.58*t + 0.12*kinetic
			g = 0.45 + 0.48*t + 0.18*pulse
			b = 0.18 + 0.26*t
		} else {
			r = 0.12 + 0.25*t
			g = 0.36 + 0.56*t + 0.14*pulse
			b = 0.32 + 0.64*t + 0.14*kinetic
		}
	}

	r = clamp01(r * fade)
	g = clamp01(g * fade)
	b = clamp01(b * fade)
	return color.RGBA{R: uint8(r*255 + 0.5), G: uint8(g*255 + 0.5), B: uint8(b*255 + 0.5), A: 220}
}

func addTrail(p *particle) {
	p.trail[p.next] = p.pos
	p.next++
	if p.next >= len(p.trail) {
		p.next = 0
		p.full = true
	}
}

func trailCount(p *particle) int {
	if p.full {
		return len(p.trail)
	}
	return p.next
}

func trailAt(p *particle, i int) vec2 {
	if !p.full {
		return p.trail[i]
	}
	idx := p.next + i
	if idx >= len(p.trail) {
		idx -= len(p.trail)
	}
	return p.trail[idx]
}

func fillBackground(img *image.RGBA, c color.RGBA) {
	b := img.Bounds()
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			img.SetRGBA(x, y, c)
		}
	}
}

func drawDot(img *image.RGBA, x, y, r int, c color.RGBA) {
	b := img.Bounds()
	for dy := -r; dy <= r; dy++ {
		for dx := -r; dx <= r; dx++ {
			if dx*dx+dy*dy > r*r {
				continue
			}
			px := x + dx
			py := y + dy
			if px < b.Min.X || py < b.Min.Y || px >= b.Max.X || py >= b.Max.Y {
				continue
			}
			blendPixel(img, px, py, c)
		}
	}
}

func drawLine(img *image.RGBA, x0, y0, x1, y1 int, c color.RGBA) {
	dx := absInt(x1 - x0)
	sx := -1
	if x0 < x1 {
		sx = 1
	}
	dy := -absInt(y1 - y0)
	sy := -1
	if y0 < y1 {
		sy = 1
	}
	err := dx + dy

	for {
		if x0 >= img.Bounds().Min.X && y0 >= img.Bounds().Min.Y && x0 < img.Bounds().Max.X && y0 < img.Bounds().Max.Y {
			blendPixel(img, x0, y0, c)
		}
		if x0 == x1 && y0 == y1 {
			break
		}
		e2 := 2 * err
		if e2 >= dy {
			err += dy
			x0 += sx
		}
		if e2 <= dx {
			err += dx
			y0 += sy
		}
	}
}

func blendPixel(img *image.RGBA, x, y int, c color.RGBA) {
	o := img.PixOffset(x, y)
	a := float64(c.A) / 255.0
	ia := 1 - a
	img.Pix[o+0] = uint8(float64(c.R)*a + float64(img.Pix[o+0])*ia + 0.5)
	img.Pix[o+1] = uint8(float64(c.G)*a + float64(img.Pix[o+1])*ia + 0.5)
	img.Pix[o+2] = uint8(float64(c.B)*a + float64(img.Pix[o+2])*ia + 0.5)
	img.Pix[o+3] = 255
}

func absInt(v int) int {
	if v < 0 {
		return -v
	}
	return v
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


























































































































































































































































































































































































































































































































































