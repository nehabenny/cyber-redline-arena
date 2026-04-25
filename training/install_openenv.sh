#!/bin/bash
pip3 install -q openenv gymnasium openai
python3 -c "import openenv; print('openenv OK')"
python3 -c "import gymnasium; print('gymnasium OK')"
echo "DEPS_DONE"
