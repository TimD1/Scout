PACKAGE  ?= scout
MAJOR    ?= $(shell awk -F"['.]" '/^__version__/{print $$2}' scout/__init__.py)
MINOR    ?= $(shell awk -F"['.]" '/^__version__/{print $$3}' scout/__init__.py)
SUB      ?= $(shell awk -F"['.]" '/^__version__/{print $$4}' scout/__init__.py)
PATCH    ?= 1
CODENAME ?= $(shell awk -F= '/CODENAME/{print $$2}' /etc/lsb-release)
VERSION   = $(MAJOR).$(MINOR).$(SUB)-$(PATCH)

INSTALL_DIR  := /opt/timd1/scout
INSTALL_VENV := $(INSTALL_DIR)/venv/
TORCH_VERSION = 1.1.0

define DEB_CONTROL
Package: $(PACKAGE)
	Version: $(MAJOR).$(MINOR).$(SUB)-$(PATCH)~$(CODENAME)
Priority: optional
	Section: science
Architecture: amd64
	Depends: python3
Maintainer: Tim Dunn <me.timd1@gmail.com>
	Description: Pileup Position Picker
	endef
	export DEB_CONTROL

clean:
	rm -rf build deb_dist dist scout*.tar.gz scout.egg-info *~ *.deb archive
	find . -name "*~" -delete
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete

deb: clean
	touch tmp
	rm -rf tmp

mkdir -p tmp/DEBIAN
echo "$$DEB_CONTROL" > tmp/DEBIAN/control
echo "#!/bin/bash\nln -fs $(INSTALL_VENV)/bin/scout /usr/local/bin" > tmp/DEBIAN/postinst
echo "#!/bin/bash\nrm -f /usr/local/bin/scout" > tmp/DEBIAN/postrm

mkdir -p tmp$(INSTALL_DIR)
python3 setup.py sdist
. tmp$(INSTALL_VENV)bin/activate
pip install dist/$(PACKAGE)-$(MAJOR).$(MINOR).$(SUB).tar.gz
cd tmp
fakeroot dpkg -b . ../$(PACKAGE)_$(MAJOR).$(MINOR).$(SUB)-$(PATCH).deb
rm -rf tmp
