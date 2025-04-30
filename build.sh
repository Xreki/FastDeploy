if [[ ! -d output ]]; then
	mkdir -p output/
fi

cp -r fastdeploy setup.py requirements.txt output
