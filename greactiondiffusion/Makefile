APP_NAME=greactiondiffusion
PREFIX?=$(HOME)
BINDIR?=$(PREFIX)/bin
DESTDIR?=

all: build

build:
	go mod tidy
	go build -ldflags="-s -w" -o $(APP_NAME) main.go

install: build
	mkdir -p $(DESTDIR)$(BINDIR)
	install -m 0755 $(APP_NAME) $(DESTDIR)$(BINDIR)/$(APP_NAME)

run: build
	./$(APP_NAME) -preset=coral -palette=twilight -spm=720

clean:
	rm -f $(APP_NAME)
