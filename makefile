
build:
	npm run build
	-rm -rf docs
	cp -ra public docs
	sed -i -e "s/'\//\'/" docs/index.html

