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
	"syscall"
	"time"
	"unsafe"

	"github.com/fogleman/gg"
)

const maxDepth = 10

type Branch struct {
	x1, y1, x2, y2 float64
	thickness      float64
	isLeaf         bool
	angle          float64
	color          color.Color
	depth          int
}

type Theme struct {
	R, G, B, Variance int
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
	growthRate := flag.Int("rate", 50, "Growth rate: Number of branches drawn per frame")
	pauseSec := flag.Int("pause", 4, "Seconds to pause before generating a new tree")
	frameStride := flag.Int("frame-stride", 2, "Render one frame every N growth steps (higher = lower CPU)")
	flag.Parse()

	if *frameStride < 1 {
		*frameStride = 1
	}

	c := make(chan os.Signal, 1)
	signal.Notify(c, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-c
		fmt.Print("\033_Ga=d,d=A,q=2\033\\") // Purge all images quietly
		fmt.Print("\033[?25h")               // Show cursor
		fmt.Print("\033[2J\033[H")           // Clear screen and reset cursor
		os.Exit(0)
	}()

	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	fmt.Print("\033[?25l") // Hide terminal cursor

	for {
		fmt.Print("\033[2J")
		fmt.Print("\033_Ga=d,d=A,q=2\033\\") // Purge all images quietly

		width, height := getTermPixels()
		dc := gg.NewContext(width, height)

		dc.SetRGBA(0, 0, 0, 0)
		dc.Clear()

		scale := float64(height) / 700.0
		if scale < 0.2 {
			scale = 0.2
		}

		branches := make([]Branch, 0, 4096)
		startX := float64(width / 2)
		startY := float64(height) - (40 * scale)

		dc.SetHexColor("#4A3F35")
		potW, potH := 150.0*scale, 40.0*scale
		dc.DrawRoundedRectangle(startX-(potW/2), startY, potW, potH, 8*scale)
		dc.Fill()

		theme := getSeasonalTheme()
		treeDensity := 0.2 + r.Float64()*0.8

		buildBranch(&branches, r, startX, startY, -math.Pi/2, 120*scale, 18*scale, 0, theme, scale, treeDensity)

		sort.Slice(branches, func(i, j int) bool {
			return branches[i].depth < branches[j].depth
		})

		// Setup Double Buffering IDs
		currentID := 1
		previousID := 2

		dc.SetLineCapRound()
		for i, step := 0, 0; i < len(branches); i, step = i+*growthRate, step+1 {
			end := i + *growthRate
			if end > len(branches) {
				end = len(branches)
			}

			for _, b := range branches[i:end] {
				if b.isLeaf {
					drawRealisticLeaf(dc, b)
				} else {
					dc.SetColor(b.color)
					dc.DrawLine(b.x1, b.y1, b.x2, b.y2)
					dc.SetLineWidth(b.thickness)
					dc.Stroke()
				}
			}

			shouldRender := step%*frameStride == 0 || end == len(branches)
			if !shouldRender {
				continue
			}

			// Reset cursor
			fmt.Print("\033[H")

			// Draw new frame with current ID (quietly)
			printKittyImage(dc.Image(), currentID)

			// Delete old frame behind it using previous ID (quietly)
			fmt.Printf("\033_Ga=d,d=i,q=2,i=%d\033\\", previousID)

			// Swap IDs
			currentID, previousID = previousID, currentID

			time.Sleep(120 * time.Millisecond)
		}

		time.Sleep(time.Duration(*pauseSec) * time.Second)
	}
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
		height = 700
	}

	return width, height - 40
}

func getSeasonalTheme() Theme {
	month := time.Now().Month()
	switch month {
	case time.March, time.April, time.May:
		return Theme{R: 120, G: 200, B: 80, Variance: 50}
	case time.June, time.July, time.August:
		return Theme{R: 30, G: 140, B: 50, Variance: 30}
	case time.September, time.October, time.November:
		return Theme{R: 210, G: 90, B: 20, Variance: 70}
	default:
		return Theme{R: 200, G: 220, B: 240, Variance: 30}
	}
}

func buildBranch(branches *[]Branch, r *rand.Rand, x, y, angle, length, thickness float64, depth int, theme Theme, scale, density float64) {
	x2 := x + math.Cos(angle)*length
	y2 := y + math.Sin(angle)*length

	*branches = append(*branches, Branch{
		x1: x, y1: y, x2: x2, y2: y2,
		thickness: thickness,
		isLeaf:    false,
		color:     color.RGBA{R: 70, G: 45, B: 30, A: 255},
		depth:     depth,
	})

	if depth > 4 {
		leafChance := (0.05 + (float64(depth) * 0.02)) * density
		if r.Float64() < leafChance {
			leafAngle := angle + (r.Float64()-0.5)*math.Pi/1.5
			*branches = append(*branches, Branch{
				x1: x2, y1: y2,
				thickness: (10 + r.Float64()*12) * scale,
				angle:     leafAngle,
				isLeaf:    true,
				color:     generateColor(theme, r),
				depth:     depth + 1,
			})
		}
	}

	if depth >= maxDepth {
		maxClusters := int(math.Ceil(2.0 * density))
		clusterCount := 0
		if maxClusters > 0 {
			clusterCount = r.Intn(maxClusters + 1)
		}

		for i := 0; i < clusterCount; i++ {
			leafAngle := angle + (r.Float64()-0.5)*math.Pi
			offsetX := (r.Float64() - 0.5) * 15 * scale
			offsetY := (r.Float64() - 0.5) * 15 * scale

			*branches = append(*branches, Branch{
				x1: x2 + offsetX, y1: y2 + offsetY,
				thickness: (12 + r.Float64()*15) * scale,
				angle:     leafAngle,
				isLeaf:    true,
				color:     generateColor(theme, r),
				depth:     depth + 1,
			})
		}
		return
	}

	numChild := 1
	chance := r.Float64()
	if chance < 0.75 {
		numChild = 2
	} else if chance < 0.90 {
		numChild = 3
	}

	for i := 0; i < numChild; i++ {
		angleOffset := (r.Float64() - 0.5) * (math.Pi / 1.5)
		newAngle := angle + angleOffset
		newLength := length * (0.65 + r.Float64()*0.2)
		newThickness := thickness * 0.7
		if newThickness < 1 {
			newThickness = 1
		}
		buildBranch(branches, r, x2, y2, newAngle, newLength, newThickness, depth+1, theme, scale, density)
	}
}

func drawRealisticLeaf(dc *gg.Context, b Branch) {
	dc.Push()
	dc.Translate(b.x1, b.y1)
	dc.Rotate(b.angle)

	size := b.thickness

	dc.SetColor(b.color)
	dc.MoveTo(0, 0)
	dc.QuadraticTo(size/2, -size/2, size, 0)
	dc.QuadraticTo(size/2, size/2, 0, 0)
	dc.Fill()

	dc.SetRGBA(0, 0, 0, 0.2)
	dc.SetLineWidth(size / 15)
	dc.MoveTo(0, 0)
	dc.LineTo(size*0.8, 0)
	dc.Stroke()

	dc.Pop()
}

func generateColor(theme Theme, r *rand.Rand) color.RGBA {
	return color.RGBA{
		R: clampColor(theme.R + r.Intn(theme.Variance) - theme.Variance/2),
		G: clampColor(theme.G + r.Intn(theme.Variance) - theme.Variance/2),
		B: clampColor(theme.B + r.Intn(theme.Variance) - theme.Variance/2),
		A: uint8(190 + r.Intn(65)),
	}
}

func clampColor(v int) uint8 {
	if v < 0 {
		return 0
	}
	if v > 255 {
		return 255
	}
	return uint8(v)
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
			// Added q=2 for quiet mode
			fmt.Printf("\033_Ga=T,f=100,t=d,q=2,i=%d,m=%d;%s\033\\", id, m, chunk)
		} else {
			fmt.Printf("\033_Gm=%d;%s\033\\", m, chunk)
		}
	}
}
