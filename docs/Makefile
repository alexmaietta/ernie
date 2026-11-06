PYTHON ?= python
ROOT := $(abspath ..)

.PHONY: build serve clean

build:
	cd $(ROOT) && $(PYTHON) -m mkdocs build

serve:
	cd $(ROOT) && port=$${PORT:-$$($(PYTHON) -c 'import socket; print(next(p for p in range(8000, 8100) if socket.socket().connect_ex(("127.0.0.1", p)) != 0))')} && \
		$(PYTHON) -m mkdocs serve -a 127.0.0.1:$$port

clean:
	rm -rf $(ROOT)/site
