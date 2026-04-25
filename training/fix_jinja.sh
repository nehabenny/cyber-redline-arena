#!/bin/bash
pip3 install -q --upgrade jinja2
python3 -c "import jinja2; print('jinja2:', jinja2.__version__)"
echo "JINJA_OK"
