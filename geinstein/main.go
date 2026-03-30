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
	"sort"
	"strings"
	"syscall"
	"time"
	"unsafe"

	"github.com/fogleman/gg"
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

type Point struct {
	X float64
	Y float64
}

type Affine [6]float64

var ident = Affine{1, 0, 0, 0, 1, 0}

var framePNGEncoder = png.Encoder{CompressionLevel: png.BestSpeed}
var frameBuffer bytes.Buffer
var frameBase64Buffer []byte

type Geometry interface {
	Shape() []Point
	Collect(S Affine, level int, out *[]PlacedHat)
}

type HatTile struct {
	label string
}

func (h *HatTile) Shape() []Point {
	return hatOutline
}

func (h *HatTile) Collect(S Affine, _ int, out *[]PlacedHat) {
	poly := make([]Point, len(hatOutline))
	cx := 0.0
	cy := 0.0
	for i := range hatOutline {
		poly[i] = transPt(S, hatOutline[i])
		cx += poly[i].X
		cy += poly[i].Y
	}
	if len(poly) > 0 {
		cx /= float64(len(poly))
		cy /= float64(len(poly))
	}
	*out = append(*out, PlacedHat{Label: h.label, Poly: poly, Center: Point{cx, cy}})
}

type Child struct {
	T    Affine
	Geom Geometry
}

type MetaTile struct {
	shape    []Point
	width    float64
	children []Child
}

func (m *MetaTile) Shape() []Point {
	return m.shape
}

func (m *MetaTile) AddChild(T Affine, geom Geometry) {
	m.children = append(m.children, Child{T: T, Geom: geom})
}

func (m *MetaTile) EvalChild(n, i int) Point {
	ch := m.children[n]
	shape := ch.Geom.Shape()
	return transPt(ch.T, shape[i%len(shape)])
}

func (m *MetaTile) Collect(S Affine, level int, out *[]PlacedHat) {
	if level <= 0 {
		return
	}
	for _, ch := range m.children {
		ch.Geom.Collect(mul(S, ch.T), level-1, out)
	}
}

func (m *MetaTile) Recentre() {
	if len(m.shape) == 0 {
		return
	}
	cx := 0.0
	cy := 0.0
	for _, p := range m.shape {
		cx += p.X
		cy += p.Y
	}
	cx /= float64(len(m.shape))
	cy /= float64(len(m.shape))
	shift := Point{-cx, -cy}
	for i := range m.shape {
		m.shape[i] = padd(m.shape[i], shift)
	}
	M := ttrans(-cx, -cy)
	for i := range m.children {
		m.children[i].T = mul(M, m.children[i].T)
	}
}

type PlacedHat struct {
	Label  string
	Poly   []Point
	Center Point
	Phase  float64
	Score  float64
}

var hr3 = math.Sqrt(3) / 2

var hatOutline = []Point{
	hexPt(0, 0), hexPt(-1, -1), hexPt(0, -2), hexPt(2, -2),
	hexPt(2, -1), hexPt(4, -2), hexPt(5, -1), hexPt(4, 0),
	hexPt(3, 0), hexPt(2, 2), hexPt(0, 3), hexPt(0, 2),
	hexPt(-1, 2),
}

func main() {
	spm := flag.Int("spm", 360, "Animation steps per minute")
	tileSize := flag.Float64("tile-size", 0, "Monotile size in pixels (<=0: auto from terminal size)")
	fillRate := flag.Int("fill-rate", 16, "Maximum new tiles added per simulation step")
	holdSec := flag.Float64("hold-sec", 4, "Seconds to hold a full view before restarting with a new pattern (0 disables)")
	outline := flag.Float64("outline", 1.0, "Tile outline width in pixels")
	paletteName := flag.String("palette", "twilight", "Color palette: twilight|fire|ice|forest|mono|viridis")
	engineName := flag.String("engine", "auto", "Computation engine: auto|cpu|gpu")
	randomizeSec := flag.Float64("randomize-sec", 0, "Regenerate with a new random pattern every N seconds (0 disables)")
	frameStride := flag.Int("frame-stride", 1, "Render one frame every N animation steps")
	flag.Parse()

	if *spm < 1 {
		*spm = 1
	}
	if *tileSize > 0 && *tileSize < 12 {
		*tileSize = 12
	}
	if *outline < 0 {
		*outline = 0
	}
	if *fillRate < 1 {
		*fillRate = 1
	}
	if *fillRate > 400 {
		*fillRate = 400
	}
	if *frameStride < 1 {
		*frameStride = 1
	}

	palette := parsePalette(*paletteName)
	engine := parseRenderEngine(*engineName)
	if engine == renderEngineGPU {
		fmt.Fprintln(os.Stderr, "GPU engine not available for geinstein yet, falling back to CPU")
	}

	rnd := rand.New(rand.NewSource(time.Now().UnixNano()))
	fieldSeed := rnd.Int63()
	resolveTileSize := func(w, h int) float64 {
		if *tileSize > 0 {
			return *tileSize
		}
		return autoTileSizeForViewport(w, h)
	}
	w0, h0 := getTermPixels()
	effectiveTileSize := resolveTileSize(w0, h0)
	tiles, bounds := buildEinsteinField(fieldSeed, w0, h0, effectiveTileSize)
	visibleTarget := len(tiles)
	visibleNow := 0
	if visibleTarget > 0 {
		visibleNow = 1
	}
	lastRandomize := time.Now()
	fullSince := time.Time{}

	resetPattern := func() {
		fieldSeed = rnd.Int63()
		w, h := getTermPixels()
		effectiveTileSize = resolveTileSize(w, h)
		tiles, bounds = buildEinsteinField(fieldSeed, w, h, effectiveTileSize)
		visibleTarget = len(tiles)
		visibleNow = 0
		if visibleTarget > 0 {
			visibleNow = 1
		}
		lastRandomize = time.Now()
		fullSince = time.Time{}
	}

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
	frame := int64(0)
	currentID := 1
	previousID := 2
	start := time.Now()

	render := func() {
		w, h := getTermPixels()
		if w < 32 || h < 32 {
			return
		}
		newTileSize := resolveTileSize(w, h)
		tileSizeChanged := math.Abs(newTileSize-effectiveTileSize) > 0.5
		effectiveTileSize = newTileSize
		if tileSizeChanged || !coversViewport(bounds, w, h, effectiveTileSize) {
			oldVisible := visibleNow
			tiles, bounds = buildEinsteinField(fieldSeed, w, h, effectiveTileSize)
			visibleTarget = len(tiles)
			if visibleTarget > 0 {
				if oldVisible < 1 {
					visibleNow = 1
				} else if oldVisible > visibleTarget {
					visibleNow = visibleTarget
				} else {
					visibleNow = oldVisible
				}
			} else {
				visibleNow = 0
			}
		}
		img := renderFrame(w, h, tiles, visibleNow, palette, effectiveTileSize, *outline, time.Since(start).Seconds())
		fmt.Print("\033[H")
		printKittyImage(img, currentID)
		fmt.Printf("\033_Ga=d,d=i,q=2,i=%d\033\\", previousID)
		currentID, previousID = previousID, currentID
	}

	render()
	for {
		select {
		case <-ticker.C:
			if visibleNow < visibleTarget {
				visibleNow += *fillRate
				if visibleNow > visibleTarget {
					visibleNow = visibleTarget
				}
			}

			if visibleNow < visibleTarget {
				fullSince = time.Time{}
			} else if visibleTarget > 0 && *holdSec > 0 {
				if fullSince.IsZero() {
					fullSince = time.Now()
				} else if time.Since(fullSince).Seconds() >= *holdSec {
					resetPattern()
				}
			}

			if *randomizeSec > 0 && time.Since(lastRandomize).Seconds() >= *randomizeSec {
				resetPattern()
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

func buildEinsteinField(seed int64, viewW, viewH int, tileSize float64) ([]PlacedHat, [4]float64) {
	const minLevel = 2
	const maxLevel = 6

	var bestTiles []PlacedHat
	bestBounds := [4]float64{-1, 1, -1, 1}

	for level := minLevel; level <= maxLevel; level++ {
		tiles, bounds := buildEinsteinFieldAtLevel(seed, level)
		bestTiles = tiles
		bestBounds = bounds
		if coversViewport(bounds, viewW, viewH, tileSize) {
			return tiles, bounds
		}
	}

	return bestTiles, bestBounds
}

func buildEinsteinFieldAtLevel(seed int64, level int) ([]PlacedHat, [4]float64) {
	if level < 1 {
		level = 1
	}

	H0, T0, P0, F0 := initBaseMetaTiles()
	H, T, P, F := H0, T0, P0, F0
	var patch *MetaTile
	for i := 0; i < level; i++ {
		patch = constructPatch(H, T, P, F)
		H, T, P, F = constructMetatiles(patch)
	}

	var hats []PlacedHat
	patch.Collect(ident, level+1, &hats)
	if len(hats) == 0 {
		return hats, [4]float64{-1, 1, -1, 1}
	}

	cx := 0.0
	cy := 0.0
	for i := range hats {
		cx += hats[i].Center.X
		cy += hats[i].Center.Y
	}
	cx /= float64(len(hats))
	cy /= float64(len(hats))
	for i := range hats {
		hats[i].Center.X -= cx
		hats[i].Center.Y -= cy
		for j := range hats[i].Poly {
			hats[i].Poly[j].X -= cx
			hats[i].Poly[j].Y -= cy
		}
	}

	rnd := rand.New(rand.NewSource(seed))
	for i := range hats {
		r := math.Hypot(hats[i].Center.X, hats[i].Center.Y)
		n := deterministicNoise(hats[i].Center.X, hats[i].Center.Y) + 0.2*rnd.Float64()
		hats[i].Phase = math.Mod(n*1.73+float64(i)*0.037, 1.0)
		hats[i].Score = r + 0.12*n
	}

	sort.Slice(hats, func(i, j int) bool {
		if hats[i].Score == hats[j].Score {
			return hats[i].Phase < hats[j].Phase
		}
		return hats[i].Score < hats[j].Score
	})

	minX, minY := math.Inf(1), math.Inf(1)
	maxX, maxY := math.Inf(-1), math.Inf(-1)
	for _, h := range hats {
		for _, p := range h.Poly {
			if p.X < minX {
				minX = p.X
			}
			if p.X > maxX {
				maxX = p.X
			}
			if p.Y < minY {
				minY = p.Y
			}
			if p.Y > maxY {
				maxY = p.Y
			}
		}
	}
	return hats, [4]float64{minX, maxX, minY, maxY}
}

func coversViewport(bounds [4]float64, viewW, viewH int, tileSize float64) bool {
	if viewW <= 0 || viewH <= 0 {
		return true
	}
	boundsW := bounds[1] - bounds[0]
	boundsH := bounds[3] - bounds[2]
	if boundsW <= 0 || boundsH <= 0 {
		return false
	}
	hatSpan := hatWidth()
	scale := tileSize / hatSpan
	if scale < 1 {
		scale = 1
	}
	return boundsW*scale >= float64(viewW)*1.05 && boundsH*scale >= float64(viewH)*1.05
}

func autoTileSizeForViewport(w, h int) float64 {
	if w < 1 {
		w = 1
	}
	if h < 1 {
		h = 1
	}
	short := w
	if h < short {
		short = h
	}
	s := float64(short) * 0.07
	if s < 18 {
		s = 18
	}
	if s > 96 {
		s = 96
	}
	return s
}

func renderFrame(w, h int, tiles []PlacedHat, visible int, palette paletteType, tileSize, outline, tSec float64) image.Image {
	dc := gg.NewContext(w, h)
	dc.SetRGB(0.02, 0.03, 0.05)
	dc.Clear()

	drawBackdrop(dc, tSec)

	if len(tiles) == 0 || visible <= 0 {
		return dc.Image()
	}

	hatSpan := hatWidth()
	scale := tileSize / hatSpan
	if scale < 1 {
		scale = 1
	}

	camX := 0.0
	camY := 0.0

	if visible > len(tiles) {
		visible = len(tiles)
	}
	for i := 0; i < visible; i++ {
		tile := tiles[i]
		shadeT := tile.Phase
		col := paletteColor(palette, shadeT)
		outlineCol := darken(col, 0.42)

		first := true
		minPX := math.Inf(1)
		minPY := math.Inf(1)
		maxPX := math.Inf(-1)
		maxPY := math.Inf(-1)

		for _, p := range tile.Poly {
			x := p.X - camX
			y := p.Y - camY
			sx := float64(w)/2 + x*scale
			sy := float64(h)/2 - y*scale
			if first {
				dc.NewSubPath()
				dc.MoveTo(sx, sy)
				first = false
			} else {
				dc.LineTo(sx, sy)
			}
			if sx < minPX {
				minPX = sx
			}
			if sx > maxPX {
				maxPX = sx
			}
			if sy < minPY {
				minPY = sy
			}
			if sy > maxPY {
				maxPY = sy
			}
		}
		dc.ClosePath()

		if maxPX < -32 || maxPY < -32 || minPX > float64(w+32) || minPY > float64(h+32) {
			continue
		}

		dc.SetRGBA255(int(col.R), int(col.G), int(col.B), 214)
		dc.FillPreserve()
		if outline > 0 {
			dc.SetLineWidth(outline)
			dc.SetRGBA255(int(outlineCol.R), int(outlineCol.G), int(outlineCol.B), 235)
			dc.Stroke()
		} else {
			dc.NewSubPath()
		}
	}

	return dc.Image()
}

func initBaseMetaTiles() (*MetaTile, *MetaTile, *MetaTile, *MetaTile) {
	H1Hat := &HatTile{label: "H1"}
	HHat := &HatTile{label: "H"}
	THat := &HatTile{label: "T"}
	PHat := &HatTile{label: "P"}
	FHat := &HatTile{label: "F"}

	HOutline := []Point{pt(0, 0), pt(4, 0), pt(4.5, hr3), pt(2.5, 5*hr3), pt(1.5, 5*hr3), pt(-0.5, hr3), pt(0, 0)}
	H := &MetaTile{shape: HOutline, width: 2}
	H.AddChild(matchTwo(hatOutline[5], hatOutline[7], HOutline[5], HOutline[0]), HHat)
	H.AddChild(matchTwo(hatOutline[9], hatOutline[11], HOutline[1], HOutline[2]), HHat)
	H.AddChild(matchTwo(hatOutline[5], hatOutline[7], HOutline[3], HOutline[4]), HHat)
	H.AddChild(mul(ttrans(2.5, hr3), mul(Affine{-0.5, -hr3, 0, hr3, -0.5, 0}, Affine{0.5, 0, 0, 0, -0.5, 0})), H1Hat)

	TOutline := []Point{pt(0, 0), pt(3, 0), pt(1.5, 3*hr3)}
	T := &MetaTile{shape: TOutline, width: 2}
	T.AddChild(Affine{0.5, 0, 0.5, 0, 0.5, hr3}, THat)

	POutline := []Point{pt(0, 0), pt(4, 0), pt(3, 2*hr3), pt(-1, 2*hr3)}
	P := &MetaTile{shape: POutline, width: 2}
	P.AddChild(Affine{0.5, 0, 1.5, 0, 0.5, hr3}, PHat)
	P.AddChild(mul(ttrans(0, 2*hr3), mul(Affine{0.5, hr3, 0, -hr3, 0.5, 0}, Affine{0.5, 0, 0, 0, 0.5, 0})), PHat)

	FOutline := []Point{pt(0, 0), pt(3, 0), pt(3.5, hr3), pt(3, 2*hr3), pt(-1, 2*hr3)}
	F := &MetaTile{shape: FOutline, width: 2}
	F.AddChild(Affine{0.5, 0, 1.5, 0, 0.5, hr3}, FHat)
	F.AddChild(mul(ttrans(0, 2*hr3), mul(Affine{0.5, hr3, 0, -hr3, 0.5, 0}, Affine{0.5, 0, 0, 0, 0.5, 0})), FHat)

	return H, T, P, F
}

func constructPatch(H, T, P, F *MetaTile) *MetaTile {
	rules := [][]any{
		{"H"},
		{0, 0, "P", 2},
		{1, 0, "H", 2},
		{2, 0, "P", 2},
		{3, 0, "H", 2},
		{4, 4, "P", 2},
		{0, 4, "F", 3},
		{2, 4, "F", 3},
		{4, 1, 3, 2, "F", 0},
		{8, 3, "H", 0},
		{9, 2, "P", 0},
		{10, 2, "H", 0},
		{11, 4, "P", 2},
		{12, 0, "H", 2},
		{13, 0, "F", 3},
		{14, 2, "F", 1},
		{15, 3, "H", 4},
		{8, 2, "F", 1},
		{17, 3, "H", 0},
		{18, 2, "P", 0},
		{19, 2, "H", 2},
		{20, 4, "F", 3},
		{20, 0, "P", 2},
		{22, 0, "H", 2},
		{23, 4, "F", 3},
		{23, 0, "F", 3},
		{16, 0, "P", 2},
		{9, 4, 0, 2, "T", 2},
		{4, 0, "F", 3},
	}

	ret := &MetaTile{shape: nil, width: H.width}
	shapes := map[string]Geometry{"H": H, "T": T, "P": P, "F": F}

	for _, r := range rules {
		switch len(r) {
		case 1:
			ret.AddChild(ident, shapes[r[0].(string)])
		case 4:
			base := ret.children[asInt(r[0])]
			poly := base.Geom.Shape()
			Tbase := base.T
			edge := asInt(r[1])
			Ppt := transPt(Tbase, poly[(edge+1)%len(poly)])
			Qpt := transPt(Tbase, poly[edge])
			nshp := shapes[asString(r[2])]
			npoly := nshp.Shape()
			idx := asInt(r[3])
			ret.AddChild(matchTwo(npoly[idx], npoly[(idx+1)%len(npoly)], Ppt, Qpt), nshp)
		default:
			chP := ret.children[asInt(r[0])]
			chQ := ret.children[asInt(r[2])]
			Ppt := transPt(chQ.T, chQ.Geom.Shape()[asInt(r[3])])
			Qpt := transPt(chP.T, chP.Geom.Shape()[asInt(r[1])])
			nshp := shapes[asString(r[4])]
			npoly := nshp.Shape()
			idx := asInt(r[5])
			ret.AddChild(matchTwo(npoly[idx], npoly[(idx+1)%len(npoly)], Ppt, Qpt), nshp)
		}
	}
	return ret
}

func constructMetatiles(patch *MetaTile) (*MetaTile, *MetaTile, *MetaTile, *MetaTile) {
	PI := math.Pi

	bps1 := patch.EvalChild(8, 2)
	bps2 := patch.EvalChild(21, 2)
	rbps := transPt(rotAbout(bps1, -2.0*PI/3.0), bps2)

	p72 := patch.EvalChild(7, 2)
	p252 := patch.EvalChild(25, 2)

	llc := intersect(bps1, rbps, patch.EvalChild(6, 2), p72)
	w := psub(patch.EvalChild(6, 2), llc)

	newHOutline := []Point{llc, bps1}
	w = transPt(trot(-PI/3), w)
	newHOutline = append(newHOutline, padd(newHOutline[1], w))
	newHOutline = append(newHOutline, patch.EvalChild(14, 2))
	w = transPt(trot(-PI/3), w)
	newHOutline = append(newHOutline, psub(newHOutline[3], w))
	newHOutline = append(newHOutline, patch.EvalChild(6, 2))

	newH := &MetaTile{shape: newHOutline, width: patch.width * 2}
	for _, ch := range []int{0, 9, 16, 27, 26, 6, 1, 8, 10, 15} {
		newH.AddChild(patch.children[ch].T, patch.children[ch].Geom)
	}

	newPOutline := []Point{p72, padd(p72, psub(bps1, llc)), bps1, llc}
	newP := &MetaTile{shape: newPOutline, width: patch.width * 2}
	for _, ch := range []int{7, 2, 3, 4, 28} {
		newP.AddChild(patch.children[ch].T, patch.children[ch].Geom)
	}

	newFOutline := []Point{
		bps2, patch.EvalChild(24, 2), patch.EvalChild(25, 0),
		p252, padd(p252, psub(llc, bps1)),
	}
	newF := &MetaTile{shape: newFOutline, width: patch.width * 2}
	for _, ch := range []int{21, 20, 22, 23, 24, 25} {
		newF.AddChild(patch.children[ch].T, patch.children[ch].Geom)
	}

	AAA := newHOutline[2]
	BBB := padd(newHOutline[1], psub(newHOutline[4], newHOutline[5]))
	CCC := transPt(rotAbout(BBB, -PI/3), AAA)
	newTOutline := []Point{BBB, CCC, AAA}
	newT := &MetaTile{shape: newTOutline, width: patch.width * 2}
	newT.AddChild(patch.children[11].T, patch.children[11].Geom)

	newH.Recentre()
	newP.Recentre()
	newF.Recentre()
	newT.Recentre()

	return newH, newT, newP, newF
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
	case "viridis":
		return paletteViridis
	default:
		return paletteTwilight
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

func paletteColor(p paletteType, t float64) color.RGBA {
	t = fract(t)
	pulse := 0.5 + 0.5*math.Sin(6.28318*(t*2.0+0.2))
	glow := 0.5 + 0.5*math.Sin(6.28318*(t*5.3+0.7))

	var r, g, b float64
	switch p {
	case paletteFire:
		r = 0.15 + 0.85*t + 0.15*glow
		g = 0.05 + 0.48*t + 0.20*pulse
		b = 0.02 + 0.14*t
	case paletteIce:
		r = 0.05 + 0.24*t
		g = 0.22 + 0.52*t + 0.18*pulse
		b = 0.30 + 0.64*t + 0.14*glow
	case paletteForest:
		r = 0.05 + 0.30*t + 0.05*glow
		g = 0.12 + 0.70*t + 0.20*pulse
		b = 0.04 + 0.20*t
	case paletteMono:
		v := clamp01(0.18 + 0.74*t + 0.06*pulse)
		r, g, b = v, v, v
	case paletteViridis:
		if t < 0.5 {
			u := t * 2
			r = 0.267004 + (0.127568-0.267004)*u
			g = 0.004874 + (0.566949-0.004874)*u
			b = 0.329415 + (0.550556-0.329415)*u
		} else {
			u := (t - 0.5) * 2
			r = 0.127568 + (0.993248-0.127568)*u
			g = 0.566949 + (0.906157-0.566949)*u
			b = 0.550556 + (0.143936-0.550556)*u
		}
	default:
		// twilight
		c := hsvToRGB(fract(0.70+0.95*t), 0.78, 0.62+0.32*pulse)
		r = float64(c.R) / 255.0
		g = float64(c.G) / 255.0
		b = float64(c.B) / 255.0
	}
	return color.RGBA{R: uint8(clamp01(r)*255 + 0.5), G: uint8(clamp01(g)*255 + 0.5), B: uint8(clamp01(b)*255 + 0.5), A: 255}
}

func darken(c color.RGBA, k float64) color.RGBA {
	k = clamp01(k)
	return color.RGBA{R: uint8(float64(c.R) * k), G: uint8(float64(c.G) * k), B: uint8(float64(c.B) * k), A: c.A}
}

func drawBackdrop(dc *gg.Context, t float64) {
	w := float64(dc.Width())
	h := float64(dc.Height())
	for i := 0; i < 4; i++ {
		u := float64(i) / 3.0
		x := w*(0.2+0.6*u) + 40*math.Cos(t*0.05+u*5.2)
		y := h*(0.25+0.5*u) + 30*math.Sin(t*0.04+u*4.7)
		r := 170 + 50*math.Sin(t*0.06+u*8.0)
		grad := gg.NewRadialGradient(x, y, 0, x, y, r)
		grad.AddColorStop(0, color.RGBA{R: uint8(12 + 10*i), G: uint8(14 + 8*i), B: uint8(22 + 14*i), A: 46})
		grad.AddColorStop(1, color.RGBA{R: 2, G: 3, B: 6, A: 0})
		dc.SetFillStyle(grad)
		dc.DrawCircle(x, y, r)
		dc.Fill()
	}
}

func hatWidth() float64 {
	minX := math.Inf(1)
	maxX := math.Inf(-1)
	for _, p := range hatOutline {
		if p.X < minX {
			minX = p.X
		}
		if p.X > maxX {
			maxX = p.X
		}
	}
	if maxX-minX < 1e-9 {
		return 1
	}
	return maxX - minX
}

func pt(x, y float64) Point { return Point{x, y} }
func hexPt(x, y float64) Point {
	return pt(x+0.5*y, hr3*y)
}
func padd(p, q Point) Point { return Point{p.X + q.X, p.Y + q.Y} }
func psub(p, q Point) Point { return Point{p.X - q.X, p.Y - q.Y} }

func inv(T Affine) Affine {
	det := T[0]*T[4] - T[1]*T[3]
	return Affine{
		T[4] / det,
		-T[1] / det,
		(T[1]*T[5] - T[2]*T[4]) / det,
		-T[3] / det,
		T[0] / det,
		(T[2]*T[3] - T[0]*T[5]) / det,
	}
}

func mul(A, B Affine) Affine {
	return Affine{
		A[0]*B[0] + A[1]*B[3],
		A[0]*B[1] + A[1]*B[4],
		A[0]*B[2] + A[1]*B[5] + A[2],
		A[3]*B[0] + A[4]*B[3],
		A[3]*B[1] + A[4]*B[4],
		A[3]*B[2] + A[4]*B[5] + A[5],
	}
}

func trot(ang float64) Affine {
	c := math.Cos(ang)
	s := math.Sin(ang)
	return Affine{c, -s, 0, s, c, 0}
}

func ttrans(tx, ty float64) Affine {
	return Affine{1, 0, tx, 0, 1, ty}
}

func rotAbout(p Point, ang float64) Affine {
	return mul(ttrans(p.X, p.Y), mul(trot(ang), ttrans(-p.X, -p.Y)))
}

func transPt(M Affine, P Point) Point {
	return Point{M[0]*P.X + M[1]*P.Y + M[2], M[3]*P.X + M[4]*P.Y + M[5]}
}

func matchSeg(p, q Point) Affine {
	return Affine{q.X - p.X, p.Y - q.Y, p.X, q.Y - p.Y, q.X - p.X, p.Y}
}

func matchTwo(p1, q1, p2, q2 Point) Affine {
	return mul(matchSeg(p2, q2), inv(matchSeg(p1, q1)))
}

func intersect(p1, q1, p2, q2 Point) Point {
	d := (q2.Y-p2.Y)*(q1.X-p1.X) - (q2.X-p2.X)*(q1.Y-p1.Y)
	uA := ((q2.X-p2.X)*(p1.Y-p2.Y) - (q2.Y-p2.Y)*(p1.X-p2.X)) / d
	return Point{p1.X + uA*(q1.X-p1.X), p1.Y + uA*(q1.Y-p1.Y)}
}

func deterministicNoise(x, y float64) float64 {
	v := math.Sin(x*12.9898+y*78.233) * 43758.5453
	return fract(v)
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

func fract(v float64) float64 {
	return v - math.Floor(v)
}

func hsvToRGB(h, s, v float64) color.RGBA {
	h = fract(h)
	s = clamp01(s)
	v = clamp01(v)
	hh := h * 6
	i := int(hh)
	f := hh - float64(i)
	p := v * (1 - s)
	q := v * (1 - s*f)
	t := v * (1 - s*(1-f))
	var r, g, b float64
	switch i % 6 {
	case 0:
		r, g, b = v, t, p
	case 1:
		r, g, b = q, v, p
	case 2:
		r, g, b = p, v, t
	case 3:
		r, g, b = p, q, v
	case 4:
		r, g, b = t, p, v
	default:
		r, g, b = v, p, q
	}
	return color.RGBA{R: uint8(r*255 + 0.5), G: uint8(g*255 + 0.5), B: uint8(b*255 + 0.5), A: 255}
}

func asInt(v any) int {
	switch n := v.(type) {
	case int:
		return n
	case int64:
		return int(n)
	case float64:
		return int(n)
	default:
		return 0
	}
}

func asString(v any) string {
	if s, ok := v.(string); ok {
		return s
	}
	return ""
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
