TOYS := gbonsai glife gmandelbrot glorenz gmagnetic gbrain

PREFIX ?= $(HOME)
BINDIR ?= $(PREFIX)/bin
DESTDIR ?=

.PHONY: all build install clean

all: build

build:
	@for d in $(TOYS); do \
		$(MAKE) -C $$d build || exit $$?; \
	done

install:
	@for d in $(TOYS); do \
		$(MAKE) -C $$d install PREFIX="$(PREFIX)" BINDIR="$(BINDIR)" DESTDIR="$(DESTDIR)" || exit $$?; \
	done

clean:
	@for d in $(TOYS); do \
		$(MAKE) -C $$d clean || exit $$?; \
	done
