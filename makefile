
PROJECT = wamca2016

release:
	make -C Release $(PROJECT)

clean:
	make -C Release $(PROJECT) clean

