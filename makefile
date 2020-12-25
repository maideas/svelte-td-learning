
build:
	npm run build
	cp -ra public docs
	sed -i -e "s/'\//\'/" docs/index.html

