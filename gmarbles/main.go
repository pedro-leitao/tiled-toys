package main

import (
	"bytes"
	"encoding/base64"
	"encoding/binary"
	"flag"
	"fmt"
	"image"
	"image/png"
	"math"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"
	"unsafe"

	"github.com/rajveermalviya/go-webgpu/wgpu"
)

type Winsize struct{ Row, Col, Xpixel, Ypixel uint16 }

type renderEngine int
type paletteType int

const (
	maxSpheres    = 50
	gpuParamsSize = 96 + maxSpheres*16*7 + 16
	spheresOffset = 96
	propsOffset   = spheresOffset + maxSpheres*16
	coeffOffset   = propsOffset + maxSpheres*16
	shapeOffset   = coeffOffset + maxSpheres*16
	basisXOffset  = shapeOffset + maxSpheres*16
	basisYOffset  = basisXOffset + maxSpheres*16
	basisZOffset  = basisYOffset + maxSpheres*16
	extraOffset   = basisZOffset + maxSpheres*16
)

const (
	renderEngineAuto renderEngine = iota
	renderEngineCPU
	renderEngineGPU
)

const (
	paletteTwilight paletteType = iota
	paletteFire
	paletteIce
	paletteForest
	paletteMono
	paletteViridis
)

type vec3 struct{ x, y, z float64 }

type sphere struct {
	c            vec3
	r            float64
	kind         int
	half         vec3
	ux           vec3
	uy           vec3
	uz           vec3
	mat          int
	albedo       vec3
	specularity  float64
	translucency float64
	emission     float64
}

type ray struct {
	o vec3
	d vec3
}

type hit struct {
	t       float64
	p       vec3
	n       vec3
	mat     int
	sphereI int
}

const (
	matNone = iota
	matPlane
	matMirror
	matOpaque
	matTranslucent
)

const (
	shapeSphere = iota
	shapeCube
	shapePyramid
)

type scene struct {
	spheres              []sphere
	hasNonSpherical      bool
	lightSphereI         int
	lightSphereBrightness float64
	light                vec3
	skyLuminance         float64
	ambientStrength      float64
	exposure             float64
	fogDensity           float64
	floorReflectivity    float64
	pathSamples          int
	pathBounces          int
	frameSeed            float64
	floorMinX, floorMaxX float64
	floorMinZ, floorMaxZ float64
}

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

var framePNGEncoder = png.Encoder{CompressionLevel: png.BestSpeed}
var frameBuffer bytes.Buffer
var frameBase64Buffer []byte

func main() {
	engineName := flag.String("engine", "auto", "Render engine: auto|cpu|gpu")
	paletteName := flag.String("palette", "twilight", "Color palette: twilight|fire|ice|forest|mono|viridis")
	spm := flag.Int("spm", 180, "Frames per minute")
	rotationSpeed := flag.Float64("rotation-speed", 0.24, "Camera 360 orbit speed (radians/second)")
	cameraTiltDeg := flag.Float64("camera-tilt-deg", 8.0, "Camera upward tilt offset in degrees")
	zoom := flag.Float64("zoom", 1.0, "Camera zoom factor (>0, larger = closer)")
	fovDeg := flag.Float64("fov-deg", 58.0, "Camera field of view in degrees (lower reduces perspective distortion)")
	fogDensity := flag.Float64("fog-density", 0.010, "Distance fog density (0 disables fog)")
	interestingView := flag.Bool("interesting-view", false, "Use random interesting camera viewpoints")
	floorReflectivity := flag.Float64("floor-reflectivity", 0.08, "Floor reflectivity (0..1)")
	lightSphereBrightness := flag.Float64("light-sphere-brightness", 2.4, "Smallest sphere light brightness (0 disables)")
	spp := flag.Int("spp", 4, "GPU path tracing samples per pixel")
	bounces := flag.Int("bounces", 6, "GPU path tracing max bounces")
	spheresN := flag.Int("spheres", 5, "Number of objects (1..50)")
	flag.Parse()

	if *spm < 1 {
		*spm = 1
	}
	if *spheresN < 1 {
		*spheresN = 1
	}
	if *spheresN > maxSpheres {
		*spheresN = maxSpheres
	}
	if *rotationSpeed < 0 {
		*rotationSpeed = 0
	}
	if *fogDensity < 0 {
		*fogDensity = 0
	}
	if *floorReflectivity < 0 {
		*floorReflectivity = 0
	}
	if *floorReflectivity > 1 {
		*floorReflectivity = 1
	}
	if *lightSphereBrightness < 0 {
		*lightSphereBrightness = 0
	}
	if *lightSphereBrightness > 40 {
		*lightSphereBrightness = 40
	}
	if *spp < 1 {
		*spp = 1
	}
	if *spp > 64 {
		*spp = 64
	}
	if *bounces < 1 {
		*bounces = 1
	}
	if *bounces > 12 {
		*bounces = 12
	}
	if *zoom <= 0 {
		*zoom = 1
	}
	if *fovDeg < 20 {
		*fovDeg = 20
	}
	if *fovDeg > 120 {
		*fovDeg = 120
	}

	engine := parseRenderEngine(*engineName)
	palette := parsePalette(*paletteName)
	var gpu *gpuRenderer
	if engine != renderEngineCPU {
		r, err := newGPURenderer()
		if err != nil {
			if engine == renderEngineGPU {
				fmt.Fprintf(os.Stderr, "failed to initialize GPU renderer: %v\n", err)
				os.Exit(1)
			}
			fmt.Fprintf(os.Stderr, "GPU renderer unavailable (%v), falling back to CPU\n", err)
		} else {
			gpu = r
			fmt.Fprintln(os.Stderr, "using WebGPU compute renderer")
			defer gpu.Close()
		}
	}

	sig := make(chan os.Signal, 1)
	signal.Notify(sig, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-sig
		cleanupTerminal()
		os.Exit(0)
	}()

	fmt.Print("\033[?25l\033[2J\033[H")
	fmt.Print("\033_Ga=d,d=A,q=2\033\\")
	defer cleanupTerminal()

	currentID, previousID := 1, 2
	drawFrame := func(img image.Image) {
		fmt.Print("\033[H")
		printKittyImage(img, currentID)
		fmt.Printf("\033_Ga=d,d=i,q=2,i=%d\033\\", previousID)
		currentID, previousID = previousID, currentID
	}

	seed := float64(time.Now().UnixNano()) * 1e-9
	scn := makeScene(*spheresN, seed, palette)
	scn.fogDensity = *fogDensity
	scn.floorReflectivity = *floorReflectivity
	configureLightSphere(&scn, *lightSphereBrightness)
	scn.pathSamples = *spp
	scn.pathBounces = *bounces
	frame := int64(0)
	render := func() {
		w, h := getTermPixels()
		if w < 40 || h < 30 {
			return
		}

		scn.light, scn.skyLuminance, scn.ambientStrength, scn.exposure = lightDirectionGMT(time.Now().UTC())
		scn.frameSeed = float64(time.Now().UnixNano()) * 1e-9

		camPos, camTarget := viewpoint(frame, *rotationSpeed, *spm, *spheresN, *cameraTiltDeg, *zoom)
		if *interestingView {
			camPos, camTarget = interestingViewpoint(frame, *spm, scn, seed)
		}
		var img *image.RGBA
		if gpu != nil {
			out, err := gpu.Render(w, h, camPos, camTarget, *fovDeg, scn)
			if err == nil {
				img = out
			} else {
				fmt.Fprintf(os.Stderr, "GPU render failed (%v), using CPU\n", err)
				gpu.Close()
				gpu = nil
				img = renderCPU(w, h, camPos, camTarget, *fovDeg, scn)
			}
		} else {
			img = renderCPU(w, h, camPos, camTarget, *fovDeg, scn)
		}
		drawFrame(img)
		frame++
	}

	render()
	ticker := time.NewTicker(time.Minute / time.Duration(*spm))
	defer ticker.Stop()
	for range ticker.C {
		render()
	}
}

func makeScene(sphereCount int, seed float64, palette paletteType) scene {
	if sphereCount > maxSpheres {
		sphereCount = maxSpheres
	}
	base := make([]sphere, 0, sphereCount)
	hasNonSpherical := false
	minGap := 0.03
	centerX := 0.0
	centerZ := 0.0
	radiusMin := 1.5
	radiusMax := 5.8 + 0.20*float64(sphereCount)
	if radiusMax < radiusMin+0.5 {
		radiusMax = radiusMin + 0.5
	}
	for i := 0; i < sphereCount; i++ {
		placed := false
		for attempt := 0; attempt < 140; attempt++ {
			t := float64(i+1) / float64(maxInt(1, sphereCount))
			angle := 2*math.Pi*clamp(0.5+0.5*signedNoise(seed+0.29, float64(i)*1.11+float64(attempt)*1.71, 1.0), 0, 1)

			sizeNoise := clamp(0.5+0.5*signedNoise(seed+5.77, float64(i)*3.31+float64(attempt)*0.61, 1.0), 0, 1)
			kindNoise := clamp(0.5+0.5*signedNoise(seed+21.37, float64(i)*2.17+float64(attempt)*0.43, 1.0), 0, 1)
			if i == 0 {
				kindNoise = 0
			}
			kind := shapeSphere
			half := vec3{}
			euler := vec3{}
			r := 0.0
			switch {
			case kindNoise < 0.34:
				kind = shapeSphere
				r = mix(0.24, 1.35, math.Pow(sizeNoise, 0.48))
			case kindNoise < 0.72:
				kind = shapeCube
				h := mix(0.22, 0.84, math.Pow(sizeNoise, 0.65))
				half = vec3{h, h, h}
				euler = vec3{
					signedNoise(seed+30.1, float64(i)*3.17+float64(attempt)*1.03, 0.85),
					signedNoise(seed+30.9, float64(i)*2.13+float64(attempt)*0.67, math.Pi),
					signedNoise(seed+31.7, float64(i)*4.19+float64(attempt)*0.91, 0.85),
				}
				r = math.Sqrt(3.0) * h
			default:
				kind = shapePyramid
				w := mix(0.26, 0.82, math.Pow(sizeNoise, 0.70))
				h := mix(0.24, 0.95, math.Pow(sizeNoise, 0.58))
				half = vec3{w, h, w}
				euler = vec3{
					signedNoise(seed+32.5, float64(i)*2.87+float64(attempt)*1.19, 0.55),
					signedNoise(seed+33.3, float64(i)*2.31+float64(attempt)*0.63, math.Pi),
					signedNoise(seed+34.1, float64(i)*3.63+float64(attempt)*0.77, 0.55),
				}
				r = math.Sqrt(2*w*w + h*h)
			}
			ux, uy, uz := basisFromEuler(euler)

			orbitR := mix(radiusMin, radiusMax, math.Sqrt(t)) + signedNoise(seed+1.93, float64(i)*1.87+float64(attempt)*0.77, 1.1)
			orbitR = clamp(orbitR, math.Max(radiusMin, r+0.55), radiusMax)

			x := centerX + orbitR*math.Cos(angle) + signedNoise(seed+2.31, float64(i)*2.03+float64(attempt), 0.35)
			z := centerZ + orbitR*math.Sin(angle) + signedNoise(seed+3.17, float64(i)*2.27+float64(attempt), 0.35)
			y := objectSupportHeight(sphere{kind: kind, half: half, ux: ux, uy: uy, uz: uz, r: r}) + signedNoise(seed+4.73, float64(i)*2.77+float64(attempt), 0.10)
			if y < 0.02 {
				y = 0.02
			}

			candidate := sphere{
				c:            vec3{x, y, z},
				r:            r,
				kind:         kind,
				half:         half,
				ux:           ux,
				uy:           uy,
				uz:           uz,
				mat:          matTranslucent,
				albedo:       vec3{0.95, 0.98, 1.0},
				specularity:  clamp(0.5+0.5*signedNoise(seed+9.3, float64(i)*4.01+float64(attempt), 1.0), 0, 1),
				translucency: clamp(0.5+0.5*signedNoise(seed+13.7, float64(i)*4.73+float64(attempt), 1.0), 0, 1),
			}

			overlap := false
			for _, existing := range base {
				if length(sub(candidate.c, existing.c)) < candidate.r+existing.r+minGap {
					overlap = true
					break
				}
			}
			camPos := vec3{0.0, 1.0, 0.0}
			if length(sub(candidate.c, camPos)) < candidate.r+0.9 {
				overlap = true
			}
			if !overlap {
				base = append(base, candidate)
				placed = true
				break
			}
		}

		if !placed {
			sizeNoise := math.Abs(signedNoise(seed+17.1, float64(i)*1.37, 1.0))
			kindNoise := clamp(0.5+0.5*signedNoise(seed+35.3, float64(i)*2.41, 1.0), 0, 1)
			if i == 0 {
				kindNoise = 0
			}
			kind := shapeSphere
			half := vec3{}
			euler := vec3{}
			r := 0.48 + 0.52*sizeNoise
			switch {
			case kindNoise < 0.34:
				kind = shapeSphere
				r = mix(0.30, 1.15, sizeNoise)
			case kindNoise < 0.72:
				kind = shapeCube
				h := mix(0.26, 0.78, sizeNoise)
				half = vec3{h, h, h}
				euler = vec3{signedNoise(seed+36.1, float64(i)*2.7, 0.85), signedNoise(seed+36.9, float64(i)*1.7, math.Pi), signedNoise(seed+37.7, float64(i)*2.2, 0.85)}
				r = math.Sqrt(3.0) * h
			default:
				kind = shapePyramid
				w := mix(0.28, 0.80, sizeNoise)
				h := mix(0.28, 0.90, sizeNoise)
				half = vec3{w, h, w}
				euler = vec3{signedNoise(seed+38.5, float64(i)*2.3, 0.55), signedNoise(seed+39.3, float64(i)*1.1, math.Pi), signedNoise(seed+40.1, float64(i)*1.9, 0.55)}
				r = math.Sqrt(2*w*w + h*h)
			}
			ux, uy, uz := basisFromEuler(euler)
			a := float64(i) * 2.399963229728653
			fallbackR := radiusMin + (radiusMax-radiusMin)*float64(i+1)/float64(maxInt(1, sphereCount)) + r + 0.7
			for guard := 0; guard < 80; guard++ {
				obj := sphere{kind: kind, half: half, ux: ux, uy: uy, uz: uz, r: r}
				y := objectSupportHeight(obj) + 0.02
				c := vec3{centerX + fallbackR*math.Cos(a), y, centerZ + fallbackR*math.Sin(a)}
				overlap := false
				for _, existing := range base {
					if length(sub(c, existing.c)) < r+existing.r+minGap {
						overlap = true
						break
					}
				}
				if !overlap && length(sub(c, vec3{0, 1, 0})) >= r+0.9 {
					obj.c = c
					obj.mat = matTranslucent
					obj.albedo = vec3{0.95, 0.98, 1.0}
					obj.specularity = clamp(0.5+0.5*signedNoise(seed+9.3, float64(i)*4.01, 1.0), 0, 1)
					obj.translucency = clamp(0.5+0.5*signedNoise(seed+13.7, float64(i)*4.73, 1.0), 0, 1)
					base = append(base, obj)
					break
				}
				fallbackR += r + 0.35
				a += 0.73
			}
		}
	}

	for _, s := range base {
		if s.kind != shapeSphere {
			hasNonSpherical = true
			break
		}
	}

	opaquePalette := opaquePaletteFor(palette)

	for i := range base {
		r := math.Abs(signedNoise(seed+3.1, float64(i)*1.73, 1.0))
		switch {
		case i%3 == 0 || r < 0.28:
			base[i].mat = matMirror
			base[i].albedo = vec3{0.96, 0.97, 1.0}
		case i%3 == 1 || r < 0.62:
			base[i].mat = matOpaque
			base[i].albedo = opaquePalette[i%len(opaquePalette)]
		default:
			base[i].mat = matTranslucent
			base[i].albedo = mix3(vec3{0.82, 0.92, 1.0}, vec3{1.0, 0.90, 0.86}, math.Abs(signedNoise(seed+7.7, float64(i)*2.11, 1.0)))
		}
	}

	minX, maxX := 1e9, -1e9
	minZ, maxZ := 1e9, -1e9
	for _, s := range base {
		if s.c.x-s.r < minX {
			minX = s.c.x - s.r
		}
		if s.c.x+s.r > maxX {
			maxX = s.c.x + s.r
		}
		if s.c.z-s.r < minZ {
			minZ = s.c.z - s.r
		}
		if s.c.z+s.r > maxZ {
			maxZ = s.c.z + s.r
		}
	}
	margin := 2.4
	return scene{
		spheres:         base,
		hasNonSpherical: hasNonSpherical,
		lightSphereI:    -1,
		lightSphereBrightness: 0,
		light:           norm(vec3{-0.6, 1.0, -0.4}),
		skyLuminance:    1.0,
		ambientStrength: 1.0,
		exposure:        1.0,
		fogDensity:      0.010,
		floorReflectivity: 0.08,
		pathSamples:     4,
		pathBounces:     6,
		frameSeed:       seed,
		floorMinX:       minX - margin,
		floorMaxX:       maxX + margin,
		floorMinZ:       minZ - margin,
		floorMaxZ:       maxZ + margin,
	}
}

func configureLightSphere(scn *scene, brightness float64) {
	scn.lightSphereI = -1
	scn.lightSphereBrightness = clamp(brightness, 0, 40)
	for i := range scn.spheres {
		scn.spheres[i].emission = 0
	}
	if scn.lightSphereBrightness <= 0 {
		return
	}
	bestI := -1
	bestR := 1e30
	for i, s := range scn.spheres {
		if s.kind != shapeSphere {
			continue
		}
		if s.r < bestR {
			bestR = s.r
			bestI = i
		}
	}
	if bestI < 0 {
		return
	}
	scn.lightSphereI = bestI
	scn.spheres[bestI].emission = scn.lightSphereBrightness
	scn.spheres[bestI].albedo = mix3(scn.spheres[bestI].albedo, vec3{1.0, 0.96, 0.88}, 0.65)
}

func viewpoint(frame int64, rotationSpeed float64, spm int, sphereCount int, cameraTiltDeg float64, zoom float64) (vec3, vec3) {
	timeSec := float64(frame) * (60.0 / float64(spm))
	angle := rotationSpeed * timeSec
	_ = sphereCount
	cam := vec3{0.0, 1.0, 0.0}
	tilt := cameraTiltDeg * math.Pi / 180.0
	dir := vec3{math.Cos(tilt) * math.Cos(angle), math.Sin(tilt), math.Cos(tilt) * math.Sin(angle)}
	target := add(cam, scale(dir, 1.0/clamp(zoom, 0.1, 10.0)))
	return cam, target
}

func interestingViewpoint(frame int64, spm int, scn scene, seed float64) (vec3, vec3) {
	holdFrames := maxInt(2, spm/90)
	shot := int(frame / int64(holdFrames))
	timeSec := float64(frame) * (60.0 / float64(maxInt(1, spm)))
	center := vec3{(scn.floorMinX + scn.floorMaxX) * 0.5, 0.9, (scn.floorMinZ + scn.floorMaxZ) * 0.5}

	maxR := 1.2
	for _, s := range scn.spheres {
		d := length(sub(s.c, center)) + s.r
		if d > maxR {
			maxR = d
		}
	}

	// choose an interesting target point
	target := center
	if len(scn.spheres) > 0 {
		choice := int(clamp((0.5+0.5*signedNoise(seed+41.3, float64(shot)*1.37, 1.0))*4, 0, 3.999))
		i0 := int(clamp((0.5+0.5*signedNoise(seed+42.1, float64(shot)*1.11, 1.0))*float64(len(scn.spheres)), 0, float64(len(scn.spheres)-1)))
		i1 := int(clamp((0.5+0.5*signedNoise(seed+42.9, float64(shot)*0.93, 1.0))*float64(len(scn.spheres)), 0, float64(len(scn.spheres)-1)))
		s0 := scn.spheres[i0].c
		s1 := scn.spheres[i1].c
		switch choice {
		case 0:
			target = s0
		case 1:
			target = scale(add(s0, s1), 0.5)
		case 2:
			if scn.lightSphereI >= 0 && scn.lightSphereI < len(scn.spheres) {
				target = scn.spheres[scn.lightSphereI].c
			} else {
				target = s0
			}
		default:
			high := scn.spheres[0].c
			for _, s := range scn.spheres {
				if s.c.y > high.y {
					high = s.c
				}
			}
			target = mix3(high, center, 0.35)
		}
	}

	style := int(clamp((0.5+0.5*signedNoise(seed+46.7, float64(shot)*0.73, 1.0))*5, 0, 4.999))
	az := 2 * math.Pi * clamp(0.5+0.5*signedNoise(seed+43.7, float64(shot)*1.61, 1.0), 0, 1)
	az += 0.22 * timeSec
	elev := mix(0.14, 0.95, clamp(0.5+0.5*signedNoise(seed+44.5, float64(shot)*1.23, 1.0), 0, 1))
	dist := mix(0.95*maxR, 1.65*maxR, clamp(0.5+0.5*signedNoise(seed+45.3, float64(shot)*0.87, 1.0), 0, 1))

	// Style-specific framing: include close zoom-ins and near-ground shots.
	switch style {
	case 0: // close-up zoom
		elev = mix(0.20, 0.65, clamp(0.5+0.5*signedNoise(seed+47.1, float64(shot), 1.0), 0, 1))
		dist = mix(0.30*maxR, 0.70*maxR, clamp(0.5+0.5*signedNoise(seed+47.9, float64(shot), 1.0), 0, 1))
	case 1: // very close detail pass
		elev = mix(0.10, 0.38, clamp(0.5+0.5*signedNoise(seed+48.3, float64(shot), 1.0), 0, 1))
		dist = mix(0.22*maxR, 0.50*maxR, clamp(0.5+0.5*signedNoise(seed+49.1, float64(shot), 1.0), 0, 1))
	case 2: // near-ground cinematic
		elev = mix(0.03, 0.14, clamp(0.5+0.5*signedNoise(seed+49.7, float64(shot), 1.0), 0, 1))
		dist = mix(0.55*maxR, 1.10*maxR, clamp(0.5+0.5*signedNoise(seed+50.5, float64(shot), 1.0), 0, 1))
		target.y = clamp(target.y, 0.08, 0.75)
	case 3: // just-above-ground, wider
		elev = mix(0.05, 0.20, clamp(0.5+0.5*signedNoise(seed+51.1, float64(shot), 1.0), 0, 1))
		dist = mix(0.85*maxR, 1.35*maxR, clamp(0.5+0.5*signedNoise(seed+51.9, float64(shot), 1.0), 0, 1))
		target.y = clamp(target.y, 0.08, 0.90)
	default: // general interesting orbit
	}

	horiz := dist * math.Cos(elev)
	cam := vec3{
		target.x + horiz*math.Cos(az),
		target.y + dist*math.Sin(elev),
		target.z + horiz*math.Sin(az),
	}
	if cam.y < 0.14 {
		cam.y = 0.14
	}
	if style == 2 || style == 3 {
		cam.y = clamp(cam.y, 0.14, 0.55)
	}

	// keep camera from being too far from the scene and out of the floor bounds
	pad := 0.85 * maxR
	cam.x = clamp(cam.x, scn.floorMinX-pad, scn.floorMaxX+pad)
	cam.z = clamp(cam.z, scn.floorMinZ-pad, scn.floorMaxZ+pad)

	// ensure camera is not inside geometry
	for _, s := range scn.spheres {
		d := length(sub(cam, s.c))
		minD := s.r + 0.45
		if d < minD && d > 1e-6 {
			cam = add(s.c, scale(norm(sub(cam, s.c)), minD))
		}
	}

	target.y = math.Max(0.08, target.y)
	return cam, target
}

func lightDirectionGMT(now time.Time) (vec3, float64, float64, float64) {
	seconds := float64(now.Hour()*3600+now.Minute()*60+now.Second()) + float64(now.Nanosecond())*1e-9
	phase := 2 * math.Pi * (seconds / 86400.0)
	const sunIntensity = 1.0
	const moonIntensity = 0.16

	// Sun: highest near 12:00 GMT, below horizon near midnight.
	sunAlt := math.Sin(phase - math.Pi/2)
	if sunAlt >= 0 {
		az := phase - math.Pi/2
		x := 0.65 * math.Cos(az)
		z := 0.80 * math.Sin(az)
		y := 0.20 + 0.95*sunAlt
		skyLuma := clamp(0.45+0.55*sunAlt, 0.45, 1.0)
		ambient := mix(0.55, 1.0, skyLuma)
		exposure := 1.0
		return scale(norm(vec3{x, y, z}), sunIntensity), skyLuma, ambient, exposure
	}

	// Moon: opposite side of the sky during GMT night.
	moonAz := phase + math.Pi/2
	moonAlt := -sunAlt
	x := 0.35 * math.Cos(moonAz)
	z := 0.55 * math.Sin(moonAz)
	y := 0.12 + 0.55*moonAlt
	skyLuma := clamp(0.03+0.09*moonAlt, 0.03, 0.12)
	ambient := clamp(0.14+0.16*moonAlt, 0.14, 0.30)
	exposure := 0.82
	return scale(norm(vec3{x, y, z}), moonIntensity), skyLuma, ambient, exposure
}

func renderCPU(w, h int, camPos, camTarget vec3, fovDeg float64, scn scene) *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	fwd := norm(sub(camTarget, camPos))
	rgt := norm(cross(fwd, vec3{0, 1, 0}))
	up := cross(rgt, fwd)
	aspect := float64(w) / float64(h)
	fovScale := math.Tan(clamp(fovDeg, 20, 120) * math.Pi / 360.0)

	for y := 0; y < h; y++ {
		v := 1 - 2*(float64(y)+0.5)/float64(h)
		for x := 0; x < w; x++ {
			u := (2*(float64(x)+0.5)/float64(w) - 1) * aspect * fovScale
			vv := v * fovScale
			r := ray{o: camPos, d: norm(add(fwd, add(scale(rgt, u), scale(up, vv))))}
			c := scale(trace(r, scn, 0), scn.exposure)
			o := y*img.Stride + x*4
			img.Pix[o+0] = toByte(c.x)
			img.Pix[o+1] = toByte(c.y)
			img.Pix[o+2] = toByte(c.z)
			img.Pix[o+3] = 255
		}
	}
	return img
}

func trace(r ray, scn scene, depth int) vec3 {
	if depth > 2 {
		return skyColor(r.d, scn.skyLuminance)
	}
	h := intersectScene(r, scn)
	if h.mat == matNone {
		return skyColor(r.d, scn.skyLuminance)
	}
	var col vec3
	if h.mat == matPlane {
		col = shadePlane(h, r, scn, depth)
		return applyDistanceFog(col, h.t, r.d, scn.fogDensity, scn.skyLuminance)
	}
	if h.sphereI >= 0 {
		s := scn.spheres[h.sphereI]
		if s.emission > 0 {
			col = gamma(scale(s.albedo, s.emission))
			return applyDistanceFog(col, h.t, r.d, scn.fogDensity, scn.skyLuminance)
		}
	}
	switch h.mat {
	case matMirror:
		col = shadeMirror(h, r, scn, depth)
	case matOpaque:
		col = shadeOpaque(h, r, scn, depth)
	default:
		col = shadeTranslucent(h, r, scn, depth)
	}
	return applyDistanceFog(col, h.t, r.d, scn.fogDensity, scn.skyLuminance)
}

func shadePlane(h hit, r ray, scn scene, depth int) vec3 {
	base := floorBaseColor(h.p.x, h.p.z)
	lambert := math.Max(0, dot(h.n, scn.light))
	shadow := objectShadow(add(h.p, scale(h.n, 0.02)), scn.light, scn.spheres)
	lambert *= shadow
	amb := clamp(scn.ambientStrength, 0.12, 1.0)
	col := add(scale(base, 0.06*amb+0.95*lambert), scale(vec3{0.04, 0.07, 0.12}, 0.08*amb))
	col = add(col, emissiveSphereContribution(h.p, h.n, scn))
	reflAmt := clamp(scn.floorReflectivity, 0, 1)
	if reflAmt > 0 {
		reflDir := reflect(r.d, h.n)
		reflCol := trace(ray{o: add(h.p, scale(h.n, 0.01)), d: reflDir}, scn, depth+1)
		col = mix3(col, reflCol, reflAmt)
	}
	return gamma(col)
}

func floorBaseColor(x, z float64) vec3 {
	chk := checker(x, z)
	return mix3(vec3{0.12, 0.12, 0.13}, vec3{0.84, 0.84, 0.88}, chk)
}

func hash1(seed, x, z float64) float64 {
	v := math.Sin((x+seed*0.13)*127.1+(z-seed*0.07)*311.7+seed*17.0) * 43758.5453123
	return v - math.Floor(v)
}

func perlinGrad2(seed, ix, iz float64) vec3 {
	a := hash1(seed, ix, iz) * 2 * math.Pi
	return vec3{math.Cos(a), math.Sin(a), 0}
}

func perlin2(seed, x, z float64) float64 {
	ix := math.Floor(x)
	iz := math.Floor(z)
	fx := x - ix
	fz := z - iz
	u := fx * fx * fx * (fx*(fx*6-15) + 10)
	v := fz * fz * fz * (fz*(fz*6-15) + 10)
	g00 := perlinGrad2(seed, ix, iz)
	g10 := perlinGrad2(seed, ix+1, iz)
	g01 := perlinGrad2(seed, ix, iz+1)
	g11 := perlinGrad2(seed, ix+1, iz+1)
	d00 := g00.x*fx + g00.y*fz
	d10 := g10.x*(fx-1) + g10.y*fz
	d01 := g01.x*fx + g01.y*(fz-1)
	d11 := g11.x*(fx-1) + g11.y*(fz-1)
	x1 := mix(d00, d10, u)
	x2 := mix(d01, d11, u)
	return mix(x1, x2, v)
}

func fbmPerlin2(seed, x, z float64, octaves int, lacunarity, gain float64) float64 {
	f, a := 1.0, 0.5
	total, normW := 0.0, 0.0
	for i := 0; i < octaves; i++ {
		total += a * (0.5 + 0.5*perlin2(seed+float64(i)*11.13, x*f, z*f))
		normW += a
		f *= lacunarity
		a *= gain
	}
	if normW == 0 {
		return 0
	}
	return total / normW
}

func smoothstep(edge0, edge1, x float64) float64 {
	t := clamp((x-edge0)/(edge1-edge0), 0, 1)
	return t * t * (3 - 2*t)
}

func applyDistanceFog(c vec3, dist float64, viewDir vec3, density float64, skyLuma float64) vec3 {
	if density <= 0 {
		return c
	}
	fog := 1 - math.Exp(-density*dist*dist)
	fogSky := skyColor(viewDir, skyLuma)
	return mix3(c, fogSky, clamp(fog, 0, 1))
}

func shadeMirror(h hit, r ray, scn scene, depth int) vec3 {
	s := scn.spheres[h.sphereI]
	reflDir := reflect(r.d, h.n)
	reflCol := trace(ray{o: add(h.p, scale(h.n, 0.01)), d: reflDir}, scn, depth+1)
	refrCol := reflCol
	if rd, ok := refract(r.d, h.n, 1.0/1.5); ok {
		refrCol = trace(ray{o: add(h.p, scale(rd, 0.01)), d: rd}, scn, depth+1)
	}
	fres := math.Pow(1-clamp(-dot(r.d, h.n), 0, 1), 5)
	amb := clamp(scn.ambientStrength, 0.12, 1.0)
	base := scale(s.albedo, 0.02+0.06*amb)
	reflWeight := mix(0.25, 1.0, s.specularity)
	glassBlend := clamp(s.translucency, 0, 1)
	col := add(scale(mix3(refrCol, reflCol, reflWeight), 0.88+0.12*fres), base)
	col = mix3(col, mul(col, s.albedo), 0.35*glassBlend)
	return gamma(col)
}

func shadeOpaque(h hit, r ray, scn scene, depth int) vec3 {
	s := scn.spheres[h.sphereI]
	lambert := math.Max(0, dot(h.n, scn.light))
	shadow := objectShadow(add(h.p, scale(h.n, 0.02)), scn.light, scn.spheres)
	lambert *= shadow
	amb := clamp(scn.ambientStrength, 0.12, 1.0)
	view := scale(r.d, -1)
	halfv := norm(add(view, scn.light))
	specPow := mix(8, 96, s.specularity)
	spec := math.Pow(math.Max(0, dot(h.n, halfv)), specPow) * (0.05 + 0.95*s.specularity) * shadow
	refl := trace(ray{o: add(h.p, scale(h.n, 0.01)), d: reflect(r.d, h.n)}, scn, depth+1)
	base := add(scale(s.albedo, 0.06*amb+0.95*lambert), vec3{spec, spec, spec})
	base = add(base, emissiveSphereContribution(h.p, h.n, scn))
	base = add(base, scale(refl, 0.04+0.26*s.specularity))

	transCol := refl
	if rd, ok := refract(r.d, h.n, 1.0/1.3); ok {
		transCol = trace(ray{o: add(h.p, scale(rd, 0.01)), d: rd}, scn, depth+1)
	}
	col := mix3(base, mul(transCol, s.albedo), s.translucency)
	return gamma(col)
}

func emissiveSphereContribution(p, n vec3, scn scene) vec3 {
	i := scn.lightSphereI
	if i < 0 || i >= len(scn.spheres) {
		return vec3{}
	}
	ls := scn.spheres[i]
	if ls.emission <= 0 {
		return vec3{}
	}
	toL := sub(ls.c, p)
	d2 := dot(toL, toL)
	if d2 < 1e-6 {
		return vec3{}
	}
	ld := scale(toL, 1/math.Sqrt(d2))
	nDotL := math.Max(0, dot(n, ld))
	if nDotL <= 0 {
		return vec3{}
	}
	shadow := objectShadow(add(p, scale(n, 0.02)), ld, scn.spheres)
	atten := ls.emission / (1.0 + d2)
	return scale(ls.albedo, nDotL*shadow*atten)
}

func shadeTranslucent(h hit, r ray, scn scene, depth int) vec3 {
	s := scn.spheres[h.sphereI]
	n := h.n
	eta := 1.0 / 1.5
	cosi := clamp(-dot(r.d, n), 0, 1)
	if dot(r.d, n) > 0 {
		n = scale(n, -1)
		eta = 1.5
		cosi = clamp(-dot(r.d, n), 0, 1)
	}
	kr := schlick(cosi, 1.0, 1.5)

	reflDir := reflect(r.d, n)
	reflCol := trace(ray{o: add(h.p, scale(n, 0.01)), d: reflDir}, scn, depth+1)

	refrCol := skyColor(reflDir, scn.skyLuminance)
	if rd, ok := refract(r.d, n, eta); ok {
		refrCol = trace(ray{o: add(h.p, scale(rd, 0.01)), d: rd}, scn, depth+1)
	}

	mixed := add(scale(refrCol, 1-kr), scale(reflCol, kr))
	mixed = mul(mixed, s.albedo)
	reflWeight := mix(0.15, 0.85, s.specularity)
	glass := mix3(refrCol, mixed, s.translucency)
	return gamma(mix3(glass, reflCol, reflWeight*0.35))
}

func intersectScene(r ray, scn scene) hit {
	best := hit{t: 1e30, mat: matNone, sphereI: -1}
	if t, ok := hitPlaneY0(r); ok && t < best.t {
		p := add(r.o, scale(r.d, t))
		best = hit{t: t, p: p, n: vec3{0, 1, 0}, mat: matPlane, sphereI: -1}
	}
	for i, s := range scn.spheres {
		if t, n, ok := hitObject(r, s); ok && t < best.t {
			p := add(r.o, scale(r.d, t))
			best = hit{t: t, p: p, n: n, mat: s.mat, sphereI: i}
		}
	}
	return best
}

func hitObject(r ray, s sphere) (float64, vec3, bool) {
	switch s.kind {
	case shapeCube:
		return hitCube(r, s)
	case shapePyramid:
		return hitPyramid(r, s)
	default:
		t, ok := hitSphere(r, s)
		if !ok {
			return 0, vec3{}, false
		}
		p := add(r.o, scale(r.d, t))
		return t, norm(sub(p, s.c)), true
	}
}

func hitSphere(r ray, s sphere) (float64, bool) {
	oc := sub(r.o, s.c)
	a := dot(r.d, r.d)
	b := 2 * dot(oc, r.d)
	c := dot(oc, oc) - s.r*s.r
	disc := b*b - 4*a*c
	if disc < 0 {
		return 0, false
	}
	sq := math.Sqrt(disc)
	t0 := (-b - sq) / (2 * a)
	t1 := (-b + sq) / (2 * a)
	if t0 > 0.001 {
		return t0, true
	}
	if t1 > 0.001 {
		return t1, true
	}
	return 0, false
}

func hitPlaneY0(r ray) (float64, bool) {
	if math.Abs(r.d.y) < 1e-6 {
		return 0, false
	}
	t := -r.o.y / r.d.y
	if t > 0.001 {
		return t, true
	}
	return 0, false
}

func objectShadow(ro, rd vec3, ss []sphere) float64 {
	for _, s := range ss {
		if t, _, ok := hitObject(ray{o: ro, d: rd}, s); ok && t > 0.001 {
			return 0.32
		}
	}
	return 1.0
}

func checker(x, z float64) float64 {
	cx := int(math.Floor(x * 0.7))
	cz := int(math.Floor(z * 0.7))
	if (cx+cz)&1 == 0 {
		return 0
	}
	return 1
}

func skyColor(d vec3, skyLuma float64) vec3 {
	t := 0.5 * (d.y + 1)
	l := clamp(skyLuma, 0, 1)
	nightLo := vec3{0.003, 0.005, 0.012}
	nightHi := vec3{0.020, 0.035, 0.090}
	dayLo := vec3{0.05, 0.08, 0.13}
	dayHi := vec3{0.45, 0.60, 0.95}
	lo := mix3(nightLo, dayLo, l)
	hi := mix3(nightHi, dayHi, l)
	return mix3(lo, hi, t)
}

func reflect(i, n vec3) vec3 {
	return sub(i, scale(n, 2*dot(i, n)))
}

func refract(i, n vec3, eta float64) (vec3, bool) {
	cosi := clamp(-dot(i, n), -1, 1)
	k := 1 - eta*eta*(1-cosi*cosi)
	if k < 0 {
		return vec3{}, false
	}
	return add(scale(i, eta), scale(n, eta*cosi-math.Sqrt(k))), true
}

func schlick(cosi, etai, etat float64) float64 {
	r0 := (etai - etat) / (etai + etat)
	r0 *= r0
	return r0 + (1-r0)*math.Pow(1-cosi, 5)
}

func gamma(c vec3) vec3 {
	return vec3{math.Sqrt(clamp(c.x, 0, 1)), math.Sqrt(clamp(c.y, 0, 1)), math.Sqrt(clamp(c.z, 0, 1))}
}

func toByte(v float64) uint8 { return uint8(clamp(v, 0, 1)*255 + 0.5) }

func parseRenderEngine(s string) renderEngine {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "cpu":
		return renderEngineCPU
	case "gpu", "wgpu", "webgpu":
		return renderEngineGPU
	default:
		return renderEngineAuto
	}
}

func parsePalette(name string) paletteType {
	switch strings.ToLower(strings.TrimSpace(name)) {
	case "fire":
		return paletteFire
	case "ice":
		return paletteIce
	case "forest":
		return paletteForest
	case "mono":
		return paletteMono
	case "viridis":
		return paletteViridis
	default:
		return paletteTwilight
	}
}

func opaquePaletteFor(p paletteType) []vec3 {
	switch p {
	case paletteFire:
		return []vec3{{0.98, 0.34, 0.18}, {0.96, 0.54, 0.18}, {0.92, 0.76, 0.22}, {0.78, 0.22, 0.16}, {0.62, 0.16, 0.12}, {1.00, 0.64, 0.30}}
	case paletteIce:
		return []vec3{{0.62, 0.86, 0.96}, {0.48, 0.76, 0.94}, {0.72, 0.92, 1.00}, {0.38, 0.62, 0.90}, {0.84, 0.96, 1.00}, {0.56, 0.80, 0.98}}
	case paletteForest:
		return []vec3{{0.22, 0.52, 0.28}, {0.34, 0.66, 0.36}, {0.46, 0.76, 0.42}, {0.56, 0.44, 0.24}, {0.18, 0.42, 0.24}, {0.72, 0.82, 0.50}}
	case paletteMono:
		return []vec3{{0.20, 0.20, 0.20}, {0.35, 0.35, 0.35}, {0.50, 0.50, 0.50}, {0.65, 0.65, 0.65}, {0.80, 0.80, 0.80}, {0.92, 0.92, 0.92}}
	case paletteViridis:
		return []vec3{{0.27, 0.05, 0.33}, {0.22, 0.32, 0.55}, {0.13, 0.57, 0.55}, {0.37, 0.79, 0.38}, {0.66, 0.86, 0.20}, {0.99, 0.91, 0.14}}
	default:
		return []vec3{{0.95, 0.42, 0.40}, {0.30, 0.75, 0.42}, {0.30, 0.52, 0.92}, {0.88, 0.72, 0.28}, {0.74, 0.48, 0.88}, {0.24, 0.82, 0.82}}
	}
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
		Label:          "spheres-compute.wgsl",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: spheresComputeWGSL},
	})
	if err != nil {
		r.Close()
		return nil, err
	}
	defer module.Release()

	pipeline, err := r.device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label: "spheres-compute-pipeline",
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

	paramsBuffer, err := r.device.CreateBuffer(&wgpu.BufferDescriptor{Label: "params", Size: uint64(gpuParamsSize), Usage: wgpu.BufferUsage_Uniform | wgpu.BufferUsage_CopyDst})
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

	r.bufferSize = uint64(width*height) * 4
	out, err := r.device.CreateBuffer(&wgpu.BufferDescriptor{Label: "out", Size: r.bufferSize, Usage: wgpu.BufferUsage_Storage | wgpu.BufferUsage_CopySrc})
	if err != nil {
		return err
	}
	r.outputBuffer = out
	readback, err := r.device.CreateBuffer(&wgpu.BufferDescriptor{Label: "readback", Size: r.bufferSize, Usage: wgpu.BufferUsage_MapRead | wgpu.BufferUsage_CopyDst})
	if err != nil {
		return err
	}
	r.readbackBuffer = readback

	bind, err := r.device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Layout: r.bindGroupLayout,
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: r.paramsBuffer, Size: uint64(gpuParamsSize)},
			{Binding: 1, Buffer: r.outputBuffer, Size: wgpu.WholeSize},
		},
	})
	if err != nil {
		return err
	}
	r.bindGroup = bind
	r.width, r.height = width, height
	return nil
}

func (r *gpuRenderer) Render(width, height int, camPos, camTarget vec3, fovDeg float64, scn scene) (*image.RGBA, error) {
	if err := r.ensureBuffers(width, height); err != nil {
		return nil, err
	}

	fwd := norm(sub(camTarget, camPos))
	rgt := norm(cross(fwd, vec3{0, 1, 0}))
	up := cross(rgt, fwd)
	fovScale := math.Tan(clamp(fovDeg, 20, 120) * math.Pi / 360.0)

	raw := make([]byte, gpuParamsSize)
	binary.LittleEndian.PutUint32(raw[0:], uint32(width))
	binary.LittleEndian.PutUint32(raw[4:], uint32(height))
	binary.LittleEndian.PutUint32(raw[8:], uint32(len(scn.spheres)))
	binary.LittleEndian.PutUint32(raw[12:], uint32(scn.pathSamples))
	putVec4(raw[16:], camPos, scn.frameSeed)
	putVec4(raw[32:], fwd, fovScale)
	putVec4(raw[48:], rgt, scn.ambientStrength)
	putVec4(raw[64:], up, scn.exposure)
	putVec4(raw[80:], scn.light, scn.skyLuminance)
	for i := 0; i < maxSpheres; i++ {
		o := spheresOffset + i*16
		if i < len(scn.spheres) {
			s := scn.spheres[i]
			putVec4(raw[o:], s.c, s.r)
			putVec4(raw[propsOffset+i*16:], s.albedo, float64(s.mat))
			putVec4(raw[coeffOffset+i*16:], vec3{s.specularity, s.translucency, s.emission}, 0)
			putVec4(raw[shapeOffset+i*16:], s.half, float64(s.kind))
			putVec4(raw[basisXOffset+i*16:], s.ux, 0)
			putVec4(raw[basisYOffset+i*16:], s.uy, 0)
			putVec4(raw[basisZOffset+i*16:], s.uz, 0)
		} else {
			putVec4(raw[o:], vec3{}, 0)
			putVec4(raw[propsOffset+i*16:], vec3{}, 0)
			putVec4(raw[coeffOffset+i*16:], vec3{}, 0)
			putVec4(raw[shapeOffset+i*16:], vec3{}, 0)
			putVec4(raw[basisXOffset+i*16:], vec3{}, 0)
			putVec4(raw[basisYOffset+i*16:], vec3{}, 0)
			putVec4(raw[basisZOffset+i*16:], vec3{}, 0)
		}
	}
	putVec4(raw[extraOffset:], vec3{float64(scn.fogDensity), float64(scn.floorReflectivity), 17.0}, float64(scn.pathBounces))
	if err := r.queue.WriteBuffer(r.paramsBuffer, 0, raw); err != nil {
		return nil, err
	}

	encoder, err := r.device.CreateCommandEncoder(&wgpu.CommandEncoderDescriptor{Label: "spheres-encoder"})
	if err != nil {
		return nil, err
	}
	defer encoder.Release()
	pass := encoder.BeginComputePass(&wgpu.ComputePassDescriptor{Label: "spheres-pass"})
	pass.SetPipeline(r.pipeline)
	pass.SetBindGroup(0, r.bindGroup, nil)
	pass.DispatchWorkgroups(uint32((width+7)/8), uint32((height+7)/8), 1)
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

	mapped := false
	status := wgpu.BufferMapAsyncStatus_Success
	if err := r.readbackBuffer.MapAsync(wgpu.MapMode_Read, 0, r.bufferSize, func(s wgpu.BufferMapAsyncStatus) {
		status = s
		mapped = true
	}); err != nil {
		return nil, err
	}
	r.device.Poll(true, nil)
	if !mapped || status != wgpu.BufferMapAsyncStatus_Success {
		return nil, fmt.Errorf("gpu readback failed")
	}

	mr := r.readbackBuffer.GetMappedRange(0, uint(r.bufferSize))
	px := wgpu.FromBytes[uint32](mr)
	img := image.NewRGBA(image.Rect(0, 0, width, height))
	for i := 0; i < width*height; i++ {
		v := px[i]
		o := i * 4
		img.Pix[o+0] = uint8(v)
		img.Pix[o+1] = uint8(v >> 8)
		img.Pix[o+2] = uint8(v >> 16)
		img.Pix[o+3] = uint8(v >> 24)
	}
	r.readbackBuffer.Unmap()
	return img, nil
}

func putVec4(dst []byte, v vec3, w float64) {
	binary.LittleEndian.PutUint32(dst[0:], math.Float32bits(float32(v.x)))
	binary.LittleEndian.PutUint32(dst[4:], math.Float32bits(float32(v.y)))
	binary.LittleEndian.PutUint32(dst[8:], math.Float32bits(float32(v.z)))
	binary.LittleEndian.PutUint32(dst[12:], math.Float32bits(float32(w)))
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

func getTermPixels() (int, int) {
	ws := &Winsize{}
	_, _, err := syscall.Syscall(syscall.SYS_IOCTL, uintptr(syscall.Stdout), uintptr(syscall.TIOCGWINSZ), uintptr(unsafe.Pointer(ws)))
	w, h := int(ws.Xpixel), int(ws.Ypixel)
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
	encLen := base64.StdEncoding.EncodedLen(len(raw))
	if cap(frameBase64Buffer) < encLen {
		frameBase64Buffer = make([]byte, encLen)
	}
	encoded := frameBase64Buffer[:encLen]
	base64.StdEncoding.Encode(encoded, raw)
	for i := 0; i < len(encoded); i += 4096 {
		end := i + 4096
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

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func signedNoise(seed, k, amp float64) float64 {
	v := math.Sin(seed*1.618+k*12.9898) * 43758.5453
	f := v - math.Floor(v)
	return (f*2 - 1) * amp
}

func add(a, b vec3) vec3           { return vec3{a.x + b.x, a.y + b.y, a.z + b.z} }
func sub(a, b vec3) vec3           { return vec3{a.x - b.x, a.y - b.y, a.z - b.z} }
func scale(v vec3, s float64) vec3 { return vec3{v.x * s, v.y * s, v.z * s} }
func mul(a, b vec3) vec3           { return vec3{a.x * b.x, a.y * b.y, a.z * b.z} }
func dot(a, b vec3) float64        { return a.x*b.x + a.y*b.y + a.z*b.z }
func length(v vec3) float64        { return math.Sqrt(dot(v, v)) }
func norm(v vec3) vec3 {
	l := length(v)
	if l == 0 {
		return vec3{}
	}
	return scale(v, 1/l)
}
func cross(a, b vec3) vec3 { return vec3{a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x} }
func clamp(v, lo, hi float64) float64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
func mix(a, b, t float64) float64 { return a*(1-t) + b*t }
func mix3(a, b vec3, t float64) vec3 {
	return vec3{mix(a.x, b.x, t), mix(a.y, b.y, t), mix(a.z, b.z, t)}
}

func rotateEuler(v, e vec3) vec3 {
	cx, sx := math.Cos(e.x), math.Sin(e.x)
	cy, sy := math.Cos(e.y), math.Sin(e.y)
	cz, sz := math.Cos(e.z), math.Sin(e.z)

	v1 := vec3{v.x, cx*v.y - sx*v.z, sx*v.y + cx*v.z}
	v2 := vec3{cy*v1.x + sy*v1.z, v1.y, -sy*v1.x + cy*v1.z}
	return vec3{cz*v2.x - sz*v2.y, sz*v2.x + cz*v2.y, v2.z}
}

func basisFromEuler(e vec3) (vec3, vec3, vec3) {
	ux := norm(rotateEuler(vec3{1, 0, 0}, e))
	uy := norm(rotateEuler(vec3{0, 1, 0}, e))
	uz := norm(rotateEuler(vec3{0, 0, 1}, e))
	return ux, uy, uz
}

func worldToLocal(v, ux, uy, uz vec3) vec3 {
	return vec3{dot(v, ux), dot(v, uy), dot(v, uz)}
}

func localToWorld(v, ux, uy, uz vec3) vec3 {
	return add(add(scale(ux, v.x), scale(uy, v.y)), scale(uz, v.z))
}

func objectSupportHeight(s sphere) float64 {
	switch s.kind {
	case shapeCube:
		return math.Abs(s.ux.y)*s.half.x + math.Abs(s.uy.y)*s.half.y + math.Abs(s.uz.y)*s.half.z
	case shapePyramid:
		verts := [5]vec3{{0, s.half.y, 0}, {-s.half.x, -s.half.y, -s.half.z}, {s.half.x, -s.half.y, -s.half.z}, {s.half.x, -s.half.y, s.half.z}, {-s.half.x, -s.half.y, s.half.z}}
		minY := 1e9
		for _, v := range verts {
			wy := localToWorld(v, s.ux, s.uy, s.uz).y
			if wy < minY {
				minY = wy
			}
		}
		return -minY
	default:
		return s.r
	}
}

func hitCube(r ray, s sphere) (float64, vec3, bool) {
	ro := worldToLocal(sub(r.o, s.c), s.ux, s.uy, s.uz)
	rd := worldToLocal(r.d, s.ux, s.uy, s.uz)
	h := s.half

	tMin, tMax := -1e30, 1e30
	for axis := 0; axis < 3; axis++ {
		var roA, rdA, hA float64
		switch axis {
		case 0:
			roA, rdA, hA = ro.x, rd.x, h.x
		case 1:
			roA, rdA, hA = ro.y, rd.y, h.y
		default:
			roA, rdA, hA = ro.z, rd.z, h.z
		}
		if math.Abs(rdA) < 1e-7 {
			if roA < -hA || roA > hA {
				return 0, vec3{}, false
			}
			continue
		}
		t1 := (-hA - roA) / rdA
		t2 := (hA - roA) / rdA
		if t1 > t2 {
			t1, t2 = t2, t1
		}
		if t1 > tMin {
			tMin = t1
		}
		if t2 < tMax {
			tMax = t2
		}
		if tMin > tMax {
			return 0, vec3{}, false
		}
	}

	t := tMin
	if t < 0.001 {
		t = tMax
	}
	if t < 0.001 {
		return 0, vec3{}, false
	}
	pl := add(ro, scale(rd, t))
	localN := vec3{}
	dx := math.Abs(math.Abs(pl.x) - h.x)
	dy := math.Abs(math.Abs(pl.y) - h.y)
	dz := math.Abs(math.Abs(pl.z) - h.z)
	if dx <= dy && dx <= dz {
		localN = vec3{math.Copysign(1, pl.x), 0, 0}
	} else if dy <= dz {
		localN = vec3{0, math.Copysign(1, pl.y), 0}
	} else {
		localN = vec3{0, 0, math.Copysign(1, pl.z)}
	}
	n := norm(localToWorld(localN, s.ux, s.uy, s.uz))
	if dot(n, r.d) > 0 {
		n = scale(n, -1)
	}
	return t, n, true
}

func hitTriangle(ro, rd, a, b, c vec3) (float64, vec3, bool) {
	e1 := sub(b, a)
	e2 := sub(c, a)
	p := cross(rd, e2)
	det := dot(e1, p)
	if math.Abs(det) < 1e-8 {
		return 0, vec3{}, false
	}
	inv := 1.0 / det
	tv := sub(ro, a)
	u := dot(tv, p) * inv
	if u < 0 || u > 1 {
		return 0, vec3{}, false
	}
	q := cross(tv, e1)
	v := dot(rd, q) * inv
	if v < 0 || u+v > 1 {
		return 0, vec3{}, false
	}
	t := dot(e2, q) * inv
	if t < 0.001 {
		return 0, vec3{}, false
	}
	n := norm(cross(e1, e2))
	return t, n, true
}

func hitPyramid(r ray, s sphere) (float64, vec3, bool) {
	ro := worldToLocal(sub(r.o, s.c), s.ux, s.uy, s.uz)
	rd := worldToLocal(r.d, s.ux, s.uy, s.uz)

	a := vec3{0, s.half.y, 0}
	b0 := vec3{-s.half.x, -s.half.y, -s.half.z}
	b1 := vec3{s.half.x, -s.half.y, -s.half.z}
	b2 := vec3{s.half.x, -s.half.y, s.half.z}
	b3 := vec3{-s.half.x, -s.half.y, s.half.z}

	tris := [][3]vec3{{a, b0, b1}, {a, b1, b2}, {a, b2, b3}, {a, b3, b0}, {b0, b2, b1}, {b0, b3, b2}}
	bestT := 1e30
	bestN := vec3{}
	for _, tri := range tris {
		t, n, ok := hitTriangle(ro, rd, tri[0], tri[1], tri[2])
		if ok && t < bestT {
			bestT = t
			bestN = n
		}
	}
	if bestT >= 1e29 {
		return 0, vec3{}, false
	}
	n := norm(localToWorld(bestN, s.ux, s.uy, s.uz))
	if dot(n, r.d) > 0 {
		n = scale(n, -1)
	}
	return bestT, n, true
}

const spheresComputeWGSL = `
struct Params {
    width: u32,
    height: u32,
    sphere_count: u32,
	spp: u32,
    cam_pos: vec4<f32>,
    cam_fwd: vec4<f32>,
    cam_right: vec4<f32>,
    cam_up: vec4<f32>,
    light_dir: vec4<f32>,
	spheres: array<vec4<f32>, 50>,
	sphere_props: array<vec4<f32>, 50>,
	sphere_coeff: array<vec4<f32>, 50>,
	shape_data: array<vec4<f32>, 50>,
	basis_x: array<vec4<f32>, 50>,
	basis_y: array<vec4<f32>, 50>,
	basis_z: array<vec4<f32>, 50>,
	board_rect: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> params: Params;

@group(0) @binding(1)
var<storage, read_write> out_pixels: array<u32>;

fn clamp01(v: f32) -> f32 {
    return clamp(v, 0.0, 1.0);
}

fn checker(x: f32, z: f32) -> f32 {
    let cx = i32(floor(x * 0.7));
    let cz = i32(floor(z * 0.7));
    if ((cx + cz) & 1) == 0 {
        return 0.0;
    }
    return 1.0;
}

fn hash1(seed: f32, x: f32, z: f32) -> f32 {
	let v = sin((x + seed * 0.13) * 127.1 + (z - seed * 0.07) * 311.7 + seed * 17.0) * 43758.5453;
	return fract(v);
}

fn perlin_grad2(seed: f32, ix: f32, iz: f32) -> vec2<f32> {
	let a = hash1(seed, ix, iz) * 6.28318530718;
	return vec2<f32>(cos(a), sin(a));
}

fn perlin2(seed: f32, x: f32, z: f32) -> f32 {
	let ix = floor(x);
	let iz = floor(z);
	let fx = fract(x);
	let fz = fract(z);
	let u = fx * fx * fx * (fx * (fx * 6.0 - 15.0) + 10.0);
	let v = fz * fz * fz * (fz * (fz * 6.0 - 15.0) + 10.0);
	let g00 = perlin_grad2(seed, ix, iz);
	let g10 = perlin_grad2(seed, ix + 1.0, iz);
	let g01 = perlin_grad2(seed, ix, iz + 1.0);
	let g11 = perlin_grad2(seed, ix + 1.0, iz + 1.0);
	let d00 = dot(g00, vec2<f32>(fx, fz));
	let d10 = dot(g10, vec2<f32>(fx - 1.0, fz));
	let d01 = dot(g01, vec2<f32>(fx, fz - 1.0));
	let d11 = dot(g11, vec2<f32>(fx - 1.0, fz - 1.0));
	let x1 = mix(d00, d10, u);
	let x2 = mix(d01, d11, u);
	return mix(x1, x2, v);
}

fn fbm_perlin2(seed: f32, x: f32, z: f32, octaves: i32, lacunarity: f32, gain: f32) -> f32 {
	var f = 1.0;
	var a = 0.5;
	var sum = 0.0;
	var w = 0.0;
	for (var i: i32 = 0; i < octaves; i = i + 1) {
		sum = sum + a * (0.5 + 0.5 * perlin2(seed + f32(i) * 11.13, x * f, z * f));
		w = w + a;
		f = f * lacunarity;
		a = a * gain;
	}
	return sum / max(1e-6, w);
}

fn floor_base_color(x: f32, z: f32) -> vec3<f32> {
	let floor_kind = 0u;
	let floor_seed = params.board_rect.z;
	if floor_kind == 1u {
		let wx = x + 1.8 * perlin2(floor_seed + 0.9, x * 0.30, z * 0.30);
		let wz = z + 1.8 * perlin2(floor_seed + 1.7, x * 0.30, z * 0.30);
		let n = fbm_perlin2(floor_seed + 1.4, wx * 1.05, wz * 1.05, 6, 2.05, 0.5);
		let n2 = fbm_perlin2(floor_seed + 4.2, wx * 2.7, wz * 2.7, 4, 2.1, 0.55);
		let mica = smoothstep(0.84, 0.985, fbm_perlin2(floor_seed + 8.3, wx * 8.8, wz * 8.8, 3, 2.0, 0.5));
		let fleck = smoothstep(0.90, 0.997, clamp01(0.5 + 0.5 * perlin2(floor_seed + 11.2, wx * 14.0, wz * 14.0)));
		let base = mix(vec3<f32>(0.19, 0.19, 0.22), vec3<f32>(0.58, 0.59, 0.63), clamp01(0.68 * n + 0.32 * n2));
		let minerals = mix(base, vec3<f32>(0.10, 0.10, 0.12), 0.22 * abs(perlin2(floor_seed + 2.8, wx * 4.3, wz * 4.3)));
		return mix(minerals, vec3<f32>(0.93, 0.93, 0.95), 0.18 * mica + 0.16 * fleck);
	}
	if floor_kind == 2u {
		let warp_x = x + 2.4 * fbm_perlin2(floor_seed + 3.7, x * 0.45, z * 0.45, 4, 2.0, 0.5);
		let warp_z = z + 2.4 * fbm_perlin2(floor_seed + 4.6, x * 0.45, z * 0.45, 4, 2.0, 0.5);
		let vein_field = fbm_perlin2(floor_seed + 6.1, warp_x * 2.2, warp_z * 2.2, 6, 2.1, 0.52);
		var ridged = 1.0 - abs(2.0 * vein_field - 1.0);
		ridged = pow(clamp01(ridged), 2.8);
		let micro = smoothstep(0.82, 0.97, fbm_perlin2(floor_seed + 7.9, warp_x * 5.0, warp_z * 5.0, 3, 2.0, 0.5));
		let base_tone = clamp01(0.45 + 0.55 * fbm_perlin2(floor_seed + 9.1, warp_x * 0.8, warp_z * 0.8, 5, 2.0, 0.5));
		let base = mix(vec3<f32>(0.72, 0.73, 0.76), vec3<f32>(0.98, 0.98, 0.995), base_tone);
		let vein_col = mix(vec3<f32>(0.35, 0.36, 0.40), vec3<f32>(0.55, 0.56, 0.62), clamp01(0.5 + 0.5 * perlin2(floor_seed + 12.4, warp_x * 1.8, warp_z * 1.8)));
		let col = mix(base, vein_col, 0.58 * ridged);
		return mix(col, vec3<f32>(0.94, 0.94, 0.97), 0.10 * micro);
	}
	let chk = checker(x, z);
	return mix(vec3<f32>(0.12, 0.12, 0.13), vec3<f32>(0.84, 0.84, 0.88), chk);
}

fn sky_color(d: vec3<f32>) -> vec3<f32> {
	let t = 0.5 * (d.y + 1.0);
	let sky_luma = clamp01(params.light_dir.w);
	let night_lo = vec3<f32>(0.003, 0.005, 0.012);
	let night_hi = vec3<f32>(0.020, 0.035, 0.090);
	let day_lo = vec3<f32>(0.05, 0.08, 0.13);
	let day_hi = vec3<f32>(0.45, 0.60, 0.95);
	let lo = mix(night_lo, day_lo, sky_luma);
	let hi = mix(night_hi, day_hi, sky_luma);
	return mix(lo, hi, t);
}

fn sphere_hit(ro: vec3<f32>, rd: vec3<f32>, s: vec4<f32>) -> f32 {
    let oc = ro - s.xyz;
    let a = dot(rd, rd);
    let b = 2.0 * dot(oc, rd);
    let c = dot(oc, oc) - s.w * s.w;
    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 {
        return -1.0;
    }
    let sq = sqrt(disc);
    let t0 = (-b - sq) / (2.0 * a);
    if t0 > 0.001 {
        return t0;
    }
    let t1 = (-b + sq) / (2.0 * a);
    if t1 > 0.001 {
        return t1;
    }
    return -1.0;
}

fn world_to_local(v: vec3<f32>, ux: vec3<f32>, uy: vec3<f32>, uz: vec3<f32>) -> vec3<f32> {
	return vec3<f32>(dot(v, ux), dot(v, uy), dot(v, uz));
}

fn local_to_world(v: vec3<f32>, ux: vec3<f32>, uy: vec3<f32>, uz: vec3<f32>) -> vec3<f32> {
	return ux * v.x + uy * v.y + uz * v.z;
}

struct ObjHit {
	t: f32,
	n: vec3<f32>,
	ok: bool,
};

fn hit_cube(ro_w: vec3<f32>, rd_w: vec3<f32>, idx: u32) -> ObjHit {
	let c = params.spheres[idx].xyz;
	let h = params.shape_data[idx].xyz;
	let ux = normalize(params.basis_x[idx].xyz);
	let uy = normalize(params.basis_y[idx].xyz);
	let uz = normalize(params.basis_z[idx].xyz);
	let ro = world_to_local(ro_w - c, ux, uy, uz);
	let rd = world_to_local(rd_w, ux, uy, uz);

	var t_min = -1e30;
	var t_max = 1e30;
	for (var axis: i32 = 0; axis < 3; axis = axis + 1) {
		let ro_a = select(select(ro.x, ro.y, axis == 1), ro.z, axis == 2);
		let rd_a = select(select(rd.x, rd.y, axis == 1), rd.z, axis == 2);
		let h_a = select(select(h.x, h.y, axis == 1), h.z, axis == 2);
		if abs(rd_a) < 1e-7 {
			if ro_a < -h_a || ro_a > h_a {
				return ObjHit(0.0, vec3<f32>(0.0), false);
			}
			continue;
		}
		var t1 = (-h_a - ro_a) / rd_a;
		var t2 = (h_a - ro_a) / rd_a;
		if t1 > t2 {
			let tt = t1;
			t1 = t2;
			t2 = tt;
		}
		t_min = max(t_min, t1);
		t_max = min(t_max, t2);
		if t_min > t_max {
			return ObjHit(0.0, vec3<f32>(0.0), false);
		}
	}

	var t = t_min;
	if t < 0.001 {
		t = t_max;
	}
	if t < 0.001 {
		return ObjHit(0.0, vec3<f32>(0.0), false);
	}

	let pl = ro + rd * t;
	let dx = abs(abs(pl.x) - h.x);
	let dy = abs(abs(pl.y) - h.y);
	let dz = abs(abs(pl.z) - h.z);
	var local_n = vec3<f32>(0.0);
	if dx <= dy && dx <= dz {
		local_n = vec3<f32>(sign(pl.x), 0.0, 0.0);
	} else if dy <= dz {
		local_n = vec3<f32>(0.0, sign(pl.y), 0.0);
	} else {
		local_n = vec3<f32>(0.0, 0.0, sign(pl.z));
	}
	var n = normalize(local_to_world(local_n, ux, uy, uz));
	if dot(n, rd_w) > 0.0 {
		n = -n;
	}
	return ObjHit(t, n, true);
}

struct TriHit {
	t: f32,
	n: vec3<f32>,
	ok: bool,
};

fn hit_triangle(ro: vec3<f32>, rd: vec3<f32>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> TriHit {
	let e1 = b - a;
	let e2 = c - a;
	let p = cross(rd, e2);
	let det = dot(e1, p);
	if abs(det) < 1e-8 {
		return TriHit(0.0, vec3<f32>(0.0), false);
	}
	let inv = 1.0 / det;
	let tv = ro - a;
	let u = dot(tv, p) * inv;
	if u < 0.0 || u > 1.0 {
		return TriHit(0.0, vec3<f32>(0.0), false);
	}
	let q = cross(tv, e1);
	let v = dot(rd, q) * inv;
	if v < 0.0 || u + v > 1.0 {
		return TriHit(0.0, vec3<f32>(0.0), false);
	}
	let t = dot(e2, q) * inv;
	if t < 0.001 {
		return TriHit(0.0, vec3<f32>(0.0), false);
	}
	return TriHit(t, normalize(cross(e1, e2)), true);
}

fn hit_pyramid(ro_w: vec3<f32>, rd_w: vec3<f32>, idx: u32) -> ObjHit {
	let c = params.spheres[idx].xyz;
	let h = params.shape_data[idx].xyz;
	let ux = normalize(params.basis_x[idx].xyz);
	let uy = normalize(params.basis_y[idx].xyz);
	let uz = normalize(params.basis_z[idx].xyz);
	let ro = world_to_local(ro_w - c, ux, uy, uz);
	let rd = world_to_local(rd_w, ux, uy, uz);

	let a = vec3<f32>(0.0, h.y, 0.0);
	let b0 = vec3<f32>(-h.x, -h.y, -h.z);
	let b1 = vec3<f32>(h.x, -h.y, -h.z);
	let b2 = vec3<f32>(h.x, -h.y, h.z);
	let b3 = vec3<f32>(-h.x, -h.y, h.z);

	var best_t = 1e30;
	var best_n = vec3<f32>(0.0, 1.0, 0.0);

	let t0 = hit_triangle(ro, rd, a, b0, b1);
	if t0.ok && t0.t < best_t { best_t = t0.t; best_n = t0.n; }
	let t1 = hit_triangle(ro, rd, a, b1, b2);
	if t1.ok && t1.t < best_t { best_t = t1.t; best_n = t1.n; }
	let t2 = hit_triangle(ro, rd, a, b2, b3);
	if t2.ok && t2.t < best_t { best_t = t2.t; best_n = t2.n; }
	let t3 = hit_triangle(ro, rd, a, b3, b0);
	if t3.ok && t3.t < best_t { best_t = t3.t; best_n = t3.n; }
	let t4 = hit_triangle(ro, rd, b0, b2, b1);
	if t4.ok && t4.t < best_t { best_t = t4.t; best_n = t4.n; }
	let t5 = hit_triangle(ro, rd, b0, b3, b2);
	if t5.ok && t5.t < best_t { best_t = t5.t; best_n = t5.n; }

	if best_t >= 1e29 {
		return ObjHit(0.0, vec3<f32>(0.0), false);
	}
	var n = normalize(local_to_world(best_n, ux, uy, uz));
	if dot(n, rd_w) > 0.0 {
		n = -n;
	}
	return ObjHit(best_t, n, true);
}

fn hit_object(ro: vec3<f32>, rd: vec3<f32>, idx: u32) -> ObjHit {
	let kind = i32(params.shape_data[idx].w + 0.5);
	if kind == 1 {
		return hit_cube(ro, rd, idx);
	}
	if kind == 2 {
		return hit_pyramid(ro, rd, idx);
	}
	let t = sphere_hit(ro, rd, params.spheres[idx]);
	if t <= 0.0 {
		return ObjHit(0.0, vec3<f32>(0.0), false);
	}
	let p = ro + rd * t;
	let n = normalize(p - params.spheres[idx].xyz);
	return ObjHit(t, n, true);
}

struct Hit {
	t: f32,
	mat: u32,
	idx: u32,
	_pad: u32,
	p: vec3<f32>,
	n: vec3<f32>,
};

fn rand01(state: ptr<function, u32>) -> f32 {
	var s = *state;
	s = s * 1664525u + 1013904223u;
	*state = s;
	return f32(s) * (1.0 / 4294967296.0);
}

fn cosine_sample_hemisphere(n: vec3<f32>, state: ptr<function, u32>) -> vec3<f32> {
	let r1 = rand01(state);
	let r2 = rand01(state);
	let phi = 6.28318530718 * r1;
	let r = sqrt(r2);
	let x = r * cos(phi);
	let z = r * sin(phi);
	let y = sqrt(max(0.0, 1.0 - r2));

	var helper = vec3<f32>(0.0, 1.0, 0.0);
	if abs(n.y) > 0.999 {
		helper = vec3<f32>(1.0, 0.0, 0.0);
	}
	let tangent = normalize(cross(helper, n));
	let bitangent = cross(n, tangent);
	return normalize(tangent * x + n * y + bitangent * z);
}

fn scene_hit(ro: vec3<f32>, rd: vec3<f32>) -> Hit {
	var h: Hit;
	h.t = 1e20;
	h.mat = 0u;
	h.idx = 0u;
	h.p = vec3<f32>(0.0);
	h.n = vec3<f32>(0.0, 1.0, 0.0);

	if abs(rd.y) > 1e-5 {
		let t_plane = -ro.y / rd.y;
		if t_plane > 0.001 && t_plane < h.t {
			h.t = t_plane;
			h.mat = 1u;
			h.p = ro + rd * t_plane;
			h.n = vec3<f32>(0.0, 1.0, 0.0);
		}
	}

	for (var i: u32 = 0u; i < params.sphere_count; i = i + 1u) {
		let oh = hit_object(ro, rd, i);
		if oh.ok && oh.t < h.t {
			let p = ro + rd * oh.t;
			h.t = oh.t;
			h.mat = u32(params.sphere_props[i].w + 0.5);
			h.idx = i;
			h.p = p;
			h.n = oh.n;
		}
	}
	return h;
}

fn in_shadow(ro: vec3<f32>, rd: vec3<f32>) -> bool {
	for (var i: u32 = 0u; i < params.sphere_count; i = i + 1u) {
		let oh = hit_object(ro, rd, i);
		if oh.ok && oh.t > 0.001 {
			return true;
		}
	}
	return false;
}

fn max3(v: vec3<f32>) -> f32 {
	return max(v.x, max(v.y, v.z));
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
	let x = gid.x;
	let y = gid.y;
	if x >= params.width || y >= params.height {
		return;
	}

	let wf = f32(params.width);
	let hf = f32(params.height);
	let aspect = wf / max(1.0, hf);
	let fov_scale = max(0.05, params.cam_fwd.w);
	let ambient_strength = clamp(params.cam_right.w, 0.12, 1.0);
	let exposure = max(0.1, params.cam_up.w);
	let spp = max(1u, params.spp);
	let max_bounces = max(1u, u32(params.board_rect.w + 0.5));
	let fog_density = max(0.0, params.board_rect.x);
	let floor_reflectivity = clamp01(params.board_rect.y);

	var accum = vec3<f32>(0.0);
	for (var s: u32 = 0u; s < spp; s = s + 1u) {
		let frame_seed_u = u32(abs(params.cam_pos.w) * 1000000.0);
		var rng = ((x + 1u) * 1973u) ^ ((y + 1u) * 9277u) ^ ((s + 1u) * 26699u) ^ (frame_seed_u * 31847u) ^ 0x9e3779b9u;

		let jx = rand01(&rng) - 0.5;
		let jy = rand01(&rng) - 0.5;
		let uv = vec2<f32>(
			(2.0 * (f32(x) + 0.5 + jx) / wf - 1.0) * aspect * fov_scale,
			(1.0 - 2.0 * (f32(y) + 0.5 + jy) / hf) * fov_scale
		);

		var ro = params.cam_pos.xyz;
		var rd = normalize(params.cam_fwd.xyz + uv.x * params.cam_right.xyz + uv.y * params.cam_up.xyz);
		let rd0 = rd;

		var throughput = vec3<f32>(1.0);
		var radiance = vec3<f32>(0.0);
		var travel = 0.0;

		for (var bounce: u32 = 0u; bounce < max_bounces; bounce = bounce + 1u) {
			let h = scene_hit(ro, rd);
			if h.mat == 0u {
				radiance = radiance + throughput * sky_color(rd);
				break;
			}

			travel = travel + h.t;
			if h.mat == 1u {
				let albedo = floor_base_color(h.p.x, h.p.z);
				let nl = max(0.0, dot(h.n, params.light_dir.xyz));
				if nl > 0.0 && !in_shadow(h.p + h.n * 0.02, params.light_dir.xyz) {
					radiance = radiance + throughput * albedo * (0.10 * ambient_strength + 0.40 * nl);
				}
				let refl = reflect(rd, h.n);
				let diffuse = cosine_sample_hemisphere(h.n, &rng);
				rd = normalize(mix(diffuse, refl, floor_reflectivity));
				throughput = throughput * mix(albedo, vec3<f32>(1.0), 0.65 * floor_reflectivity);
				ro = h.p + h.n * 0.01;
			} else {
				let albedo = params.sphere_props[h.idx].xyz;
				let coeff = params.sphere_coeff[h.idx].xyz;
				let specularity = clamp01(coeff.x);
				let translucency = clamp01(coeff.y);
				let emission = max(0.0, coeff.z);
				let n = h.n;

				if emission > 0.0 {
					radiance = radiance + throughput * albedo * emission;
					break;
				}

				if h.mat == 2u {
					let refl = reflect(rd, n);
					let diffuse = cosine_sample_hemisphere(n, &rng);
					rd = normalize(mix(diffuse, refl, 0.85 + 0.15 * specularity));
					ro = h.p + n * 0.01;
					throughput = throughput * albedo;
				} else if h.mat == 3u {
					let nl = max(0.0, dot(n, params.light_dir.xyz));
					if nl > 0.0 && !in_shadow(h.p + n * 0.02, params.light_dir.xyz) {
						radiance = radiance + throughput * albedo * (0.08 * ambient_strength + 0.45 * nl);
					}
					let refl = reflect(rd, n);
					let diffuse = cosine_sample_hemisphere(n, &rng);
					let choose_spec = rand01(&rng) < specularity;
					rd = normalize(select(diffuse, refl, choose_spec));
					ro = h.p + n * 0.01;
					throughput = throughput * albedo;
				} else {
					var nn = n;
					var eta = 1.0 / 1.5;
					if dot(rd, nn) > 0.0 {
						nn = -nn;
						eta = 1.5;
					}
					let refl = reflect(rd, nn);
					let refr = refract(rd, nn, eta);
					let refr_ok = length(refr) > 1e-4;
					let cosi = clamp(-dot(rd, nn), 0.0, 1.0);
					let fres = 0.04 + (1.0 - 0.04) * pow(1.0 - cosi, 5.0);
					let p_reflect = clamp01(fres + (1.0 - translucency) * 0.35 + specularity * 0.25);
					let use_reflect = rand01(&rng) < p_reflect || !refr_ok;
					rd = normalize(select(refr, refl, use_reflect));
					ro = h.p + rd * 0.01;
					throughput = throughput * mix(vec3<f32>(1.0), albedo, 0.75);
				}
			}

			if bounce > 2u {
				let p_survive = clamp(max3(throughput), 0.15, 0.95);
				if rand01(&rng) > p_survive {
					break;
				}
				throughput = throughput / p_survive;
			}
		}

		if fog_density > 0.0 && travel > 0.0 {
			let fog = 1.0 - exp(-fog_density * travel * travel);
			radiance = mix(radiance, sky_color(rd0), clamp01(fog));
		}

		accum = accum + radiance;
	}

	var col = accum / f32(spp);
	col = col * exposure;
	col = sqrt(max(col, vec3<f32>(0.0)));
	let r = u32(clamp01(col.x) * 255.0 + 0.5);
	let g = u32(clamp01(col.y) * 255.0 + 0.5);
	let b = u32(clamp01(col.z) * 255.0 + 0.5);
	out_pixels[y * params.width + x] = r | (g << 8u) | (b << 16u) | (255u << 24u);
}
`
