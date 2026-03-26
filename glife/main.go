package main

import (
	"bytes"
	"encoding/base64"
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"math/rand"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"time"
	"unsafe"

	"github.com/fogleman/gg"
)

const historySize = 16

type Game struct {
	grid      [][]int
	width     int
	height    int
	history   []uint64
	histIdx   int
	baseColor color.RGBA
}

type Winsize struct {
	Row    uint16
	Col    uint16
	Xpixel uint16
	Ypixel uint16
}

var framePNGEncoder = png.Encoder{CompressionLevel: png.BestSpeed}
var frameBuffer bytes.Buffer
var frameBase64Buffer []byte

func main() {
	spm := flag.Int("spm", 600, "Steps per minute (e.g., 600 SPM = 10 frames per second)")
	colorStr := flag.String("agecolour", "green", "Base color for the age gradient (e.g., red, blue, green, purple, #FF5500)")
	cellSize := flag.Int("cell-size", 10, "Cell size in pixels")
	frameStride := flag.Int("frame-stride", 1, "Render one frame every N steps (higher = lower CPU)")
	flag.Parse()

	if *spm < 1 {
		*spm = 1
	}
	if *cellSize < 2 {
		*cellSize = 2
	}
	if *frameStride < 1 {
		*frameStride = 1
	}

	baseColor := parseBaseColor(*colorStr)

	game := &Game{
		history:   make([]uint64, historySize),
		baseColor: baseColor,
	}

	wpx, hpx := getTermPixels()
	gridW := max(8, wpx/(*cellSize))
	gridH := max(8, hpx/(*cellSize))
	game.Resize(gridW, gridH)
	game.Reset()

	c := make(chan os.Signal, 1)
	signal.Notify(c, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-c
		cleanupTerminal()
		os.Exit(0)
	}()

	fmt.Print("\033[?25l")             // hide cursor
	fmt.Print("\033[2J\033[H")         // clear + home
	fmt.Print("\033_Ga=d,d=A,q=2\033\\") // purge old images
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
		wpx, hpx = getTermPixels()
		newGridW := max(8, wpx/(*cellSize))
		newGridH := max(8, hpx/(*cellSize))
		if newGridW != game.width || newGridH != game.height {
			game.Resize(newGridW, newGridH)
		}

		dc := gg.NewContext(wpx, hpx)
		game.Draw(dc, float64(*cellSize))

		fmt.Print("\033[H")
		printKittyImage(dc.Image(), currentID)
		fmt.Printf("\033_Ga=d,d=i,q=2,i=%d\033\\", previousID)
		currentID, previousID = previousID, currentID
	}

	render()

	for {
		select {
		case <-ticker.C:
			game.Update()
			step++
			if step%*frameStride == 0 {
				render()
			}
		case <-resizeTicker.C:
			render()
		}
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

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func parseBaseColor(s string) color.RGBA {
	s = strings.ToLower(strings.TrimSpace(s))
	if s == "" {
		return color.RGBA{R: 80, G: 210, B: 80, A: 255}
	}
	if strings.HasPrefix(s, "#") {
		hex := strings.TrimPrefix(s, "#")
		if len(hex) == 3 {
			hex = fmt.Sprintf("%c%c%c%c%c%c", hex[0], hex[0], hex[1], hex[1], hex[2], hex[2])
		}
		if len(hex) == 6 {
			if v, err := strconv.ParseUint(hex, 16, 32); err == nil {
				return color.RGBA{
					R: uint8((v >> 16) & 0xFF),
					G: uint8((v >> 8) & 0xFF),
					B: uint8(v & 0xFF),
					A: 255,
				}
			}
		}
	}

	named := map[string]color.RGBA{
		"red":    {R: 220, G: 70, B: 70, A: 255},
		"green":  {R: 80, G: 210, B: 80, A: 255},
		"blue":   {R: 70, G: 140, B: 230, A: 255},
		"purple": {R: 170, G: 90, B: 220, A: 255},
		"cyan":   {R: 70, G: 210, B: 210, A: 255},
		"yellow": {R: 220, G: 210, B: 80, A: 255},
		"orange": {R: 240, G: 140, B: 70, A: 255},
		"white":  {R: 240, G: 240, B: 240, A: 255},
	}
	if c, ok := named[s]; ok {
		return c
	}

	return named["green"]
}

func (g *Game) Resize(w, h int) {
	newGrid := make([][]int, h)
	for y := 0; y < h; y++ {
		newGrid[y] = make([]int, w)
		for x := 0; x < w; x++ {
			if y < g.height && x < g.width {
				newGrid[y][x] = g.grid[y][x]
			}
		}
	}
	g.grid = newGrid
	g.width = w
	g.height = h
}

func (g *Game) Reset() {
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	for y := 0; y < g.height; y++ {
		for x := 0; x < g.width; x++ {
			if r.Float32() < 0.25 {
				g.grid[y][x] = 1
			} else {
				g.grid[y][x] = 0
			}
		}
	}
	g.history = make([]uint64, historySize)
	g.histIdx = 0
}

func (g *Game) Update() {
	newGrid := make([][]int, g.height)
	survivors := 0

	for y := 0; y < g.height; y++ {
		newGrid[y] = make([]int, g.width)
		for x := 0; x < g.width; x++ {
			neighbors := g.countNeighbors(x, y)
			age := g.grid[y][x]

			if age > 0 {
				if neighbors == 2 || neighbors == 3 {
					newGrid[y][x] = age + 1
					if newGrid[y][x] > 1000 {
						newGrid[y][x] = 1000
					}
					survivors++
				} else {
					newGrid[y][x] = 0
				}
			} else {
				if neighbors == 3 {
					newGrid[y][x] = 1
					survivors++
				}
			}
		}
	}

	g.grid = newGrid

	minSurvivors := int(float64(g.width*g.height) * 0.02)
	if survivors < minSurvivors || g.isLooping() {
		g.Reset()
	}
}

func (g *Game) isLooping() bool {
	h := g.hash()
	for _, pastHash := range g.history {
		if h == pastHash && h != 0 {
			return true
		}
	}
	g.history[g.histIdx] = h
	g.histIdx = (g.histIdx + 1) % historySize
	return false
}

func (g *Game) countNeighbors(x, y int) int {
	count := 0
	for dy := -1; dy <= 1; dy++ {
		for dx := -1; dx <= 1; dx++ {
			if dx == 0 && dy == 0 {
				continue
			}
			nx := (x + dx + g.width) % g.width
			ny := (y + dy + g.height) % g.height

			if g.grid[ny][nx] > 0 {
				count++
			}
		}
	}
	return count
}

func (g *Game) hash() uint64 {
	var h uint64 = 14695981039346656037
	for y := 0; y < g.height; y++ {
		for x := 0; x < g.width; x++ {
			if g.grid[y][x] > 0 {
				h ^= uint64(y)
				h *= 1099511628211
				h ^= uint64(x)
				h *= 1099511628211
			}
		}
	}
	return h
}

func (g *Game) Draw(dc *gg.Context, cellSize float64) {
	dc.SetRGB(0.02, 0.02, 0.02)
	dc.Clear()

	pad := 1.0
	if cellSize < 4 {
		pad = 0
	}
	radius := (cellSize - pad) * 0.25

	for y := 0; y < g.height; y++ {
		for x := 0; x < g.width; x++ {
			age := g.grid[y][x]
			if age <= 0 {
				continue
			}

			c := g.getAgeColor(age)
			dc.SetColor(c)

			px := float64(x) * cellSize
			py := float64(y) * cellSize
			sz := cellSize - pad
			dc.DrawRoundedRectangle(px, py, sz, sz, radius)
			dc.Fill()
		}
	}
}

// getAgeColor dynamically calculates a gradient from white -> base color -> dark base color.
func (g *Game) getAgeColor(age int) color.RGBA {
	r := int32(g.baseColor.R)
	gr := int32(g.baseColor.G)
	b := int32(g.baseColor.B)

	if age > 100 {
		age = 100
	}

	var newR, newG, newB int32

	if age < 20 {
		// First 20 generations: Fade from White to the Base Color
		t := float64(age-1) / 19.0 // goes from 0.0 to 1.0
		newR = 255 - int32(t*float64(255-r))
		newG = 255 - int32(t*float64(255-gr))
		newB = 255 - int32(t*float64(255-b))
	} else {
		// Generations 20 to 100+: Fade from Base Color down to Dark (20% brightness)
		t := float64(age-20) / 80.0 // goes from 0.0 to 1.0
		factor := 1.0 - (t * 0.8)   // goes from 1.0 down to 0.2
		newR = int32(float64(r) * factor)
		newG = int32(float64(gr) * factor)
		newB = int32(float64(b) * factor)
	}

	return color.RGBA{R: uint8(newR), G: uint8(newG), B: uint8(newB), A: 255}
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
