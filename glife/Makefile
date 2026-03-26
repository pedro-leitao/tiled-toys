APP_NAME=glife
PREFIX?=$(HOME)
BINDIR?=$(PREFIX)/bin
DESTDIR?=

all: build

build:
	go build -ldflags="-s -w" -o $(APP_NAME) main.go

install: build
	mkdir -p $(DESTDIR)$(BINDIR)
	install -m 0755 $(APP_NAME) $(DESTDIR)$(BINDIR)/$(APP_NAME)

run: build
	./$(APP_NAME) -agecolour=cyan -spm=1000

clean:
	rm -f $(APP_NAME)
