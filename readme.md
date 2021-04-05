# Download from Github
- From the home directory in babbage, run:
```
git clone https://github.com/mbogden/babbage-project-mpi.git
```

# Create Virtual Environment
- Move into project directory and run the virtual environment script
```
cd babbage-project-mpi
./create_venv.sh
```

# Test babbage queue
- Have babbage run the qsub file to see if mpi works.
```
qsub job.qsub
```
- Use `qstat` until the job disappears
- There should now be two files with your job name in the home directory
	- test_foo.o* is the output file
	- test_foo.e* is the error file 

