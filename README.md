# Tiled Toys

<p align="center">
	<img src="images/gbonsai.png" alt="gbonsai preview" width="360" />
	<img src="images/glife.png" alt="glife preview" width="360" />
</p>
<p align="center">
	<em>gbonsai</em> — animated bonsai tree growth &nbsp;&nbsp;•&nbsp;&nbsp; <em>glife</em> — animated Conway's Game of Life
</p>

Small graphical terminal toys written in Go:

- `gbonsai`: animated bonsai tree growth
- `glife`: animated Conway's Game of Life with age-based coloring

Both are toy applications intended for **compatible terminals** that support the Kitty graphics protocol (or equivalent image escape support), such as [Ghostty](https://ghostty.org) or [Kitty](https://sw.kovidgoyal.net/kitty/). They work well as ambient visuals in **tiled window manager** layouts (e.g., [i3](https://i3wm.org/), [Hyprland](https://hyprland.org/), [Sway](https://swaywm.org/), [Awesome](https://awesomewm.org/), or [AeroSpace](https://github.com/nikitabobko/AeroSpace)).

## System requirements

- MacOS or Linux
- Go toolchain installed (project currently targets recent Go releases)
- A terminal with image protocol support used by these apps (Kitty-compatible graphics escape sequences)
- `make` and standard Unix tools (`install`, `mkdir`, etc.)

## Repository layout

- [`gbonsai/`](gbonsai/) — standalone Go module + Makefile
- [`glife/`](glife/) — standalone Go module + Makefile

## Build

Build each app from its directory:

- `cd gbonsai && make build`
- `cd glife && make build`

Or from repo root:

- `make -C gbonsai build`
- `make -C glife build`

## Install

Both Makefiles support `PREFIX`, `BINDIR`, and `DESTDIR`.

Default install (to `$HOME/bin`):

- `make -C gbonsai install`
- `make -C glife install`

Custom prefix:

- `make -C gbonsai install PREFIX=/usr/local`
- `make -C glife install PREFIX=/usr/local`

Package staging example:

- `make -C gbonsai install DESTDIR=/tmp/pkgroot PREFIX=/usr/local`
- `make -C glife install DESTDIR=/tmp/pkgroot PREFIX=/usr/local`

## Run

### gbonsai

From repo root:

- `make -C gbonsai run`

Direct binary example:

- `./gbonsai/gbonsai -pause 30 -rate 10`

Useful flags:

- `-rate` growth steps per frame
- `-pause` seconds between trees
- `-frame-stride` render every N growth steps

### glife

From repo root:

- `make -C glife run`

Direct binary example:

- `./glife/glife -agecolour=cyan -spm=1000`

Useful flags:

- `-spm` simulation steps per minute
- `-agecolour` base color (`red`, `blue`, `green`, `purple`, `#RRGGBB`, etc.)
- `-cell-size` rendered cell size in pixels
- `-frame-stride` render every N simulation steps

## Notes

- If output appears blank or garbled, use a terminal that supports the required image protocol.
- These programs are intentionally lightweight visual toys, not production TUI applications.
