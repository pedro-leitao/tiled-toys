APP_NAME=gbrain
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
	./$(APP_NAME) -nii=./average305_t1_tal_lin.nii -spm=90

clean:
	rm -f $(APP_NAME)
