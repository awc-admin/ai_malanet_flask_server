#!/bin/bash
echo "Camera Trap API"
export FLASK_APP=server
flask run --host=0.0.0.0 || { echo "Failed to start Flask."; exit 1; }