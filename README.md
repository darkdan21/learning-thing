
READ THIS FILE FULLY to understand what you have in the folder and how to do this challenge, then start at the beginning.

# Important notes

- Most scripts (\*.sh and \*.py) will respond to the '-h' option with usage instructions.
- None of the scripts are executable - you need to run the \*.sh with `sh name_of_script.sh` and python with `python3 path/to/python_script.py`.
- All of the scripts assume you are in the directory with all the \*.sh files when run (i.e. the top directory of your copy of the files we have provided).

# To train your network

Submit BlueBEAR-sbatch.sbatch as a batch job

## To save your network

The train-network.sh, which BlueBEAR-sbatch.sh calls, will save your network and trained values in a directory called 'save'.  Each full run, training your network, will create a new subfolder (named the seconds-since-UNIX-epoch when it ran -- if you don't know what this means, bigger number == more recently trained network) with a copy of the final network in it.  It is the content of **one** of these sub-folders, including all folders in it, you need to provide in your marking directory.  See [marking](#challenge-marking-and-scoring) for more information on marking.

# Testing your network with the marking sets

In the test_images directory you will find 5 sets of images:

- colour-digits: Computer generated digits which should be easily recognised
- colour-digits-fixed: Fixed version of the above that will pre-process better (i.e. use this one, of the two!)
- Ed: Sample real handwriting from 'Ed'
- Simon: Sample real handwriting from 'Simon'
- Luke: Sample real handwriting from 'Luke'

Testing the network is quick and can be done directly on the login nodes - you do not need to submit a batch job for any of these steps.

## Pre-processing

These datasets are, obviously, different from the MNIST dataset shown in the presentation (which are 28x28 pixel monochrome images).  In order to use these to test the network, we have to pre-process them using the same method used to create the MNIST dataset (as described at http://yann.lecun.com/exdb/mnist/).  We have provided a script to do this for you:

```
module load bear-apps/2018a
module load OpenCV/3.4.1-iomkl-2018a-Python-3.6.3

sh pre-process.sh -d path/to/test/images/to/process
```

**Note:** OpenCV only builds on newer processors, so will not run you the login node - you will have to write a batch job to run these commands.

## If you want to add your own training samples

Create 1 file per digit with 1:1 aspect ratio (i.e. square - currently the pre-process script will **not** pad the image to make it square) then [pre-process](#pre-processing)

## To create marking image-set for testing

After [pre-processing]((#pre-processing), the images need to be merged into a saved numpy array to use with the tensorflow command line tool.  Again, we have provided 2 scripts to do this for you:
- build-mark-file.sh: Builds a saved numpy array in 'mark.npy' with each (pre-processed) image file you give as an argument *in the order* you give them
- build-0-9-mark-file.sh: Usees 'build-mark-file.sh' to build a file with the files '0.png', '1.png', '2.png', '3.png', '4.png', '5.png', '6.png', '7.png', '8.png', '9.png' from the given directory

for example:
```
module load bear-apps/2018a
module load TensorFlow/1.8.0-foss-2018a-Python-3.6.3

sh build-0-9-mark-file.sh -d path/to/pre/processed/images
```

## Actually testing

To test, run the 'mark.sh' script with the save directory from your trained network.  It assumes that you have a `mark.npy` file generated from [making a marking dataset](to-create-marking-image-set-for-testing):

```
sh mark.sh -d save/01234567
```

# Challenge marking and scoring

Your team's '/rds/projects/2018/thompssj-bear-chal18-0X/chal5' should contain a copy of the saved network you want marked.

Your network will be marked using the same 'mark.sh' script as you have been provided with.  It will be used with 5 mark.npy files pre-generated from 5 seperate, increasingly challenging, handwritten digit samples.  You have not been given these samples so that a) this reflects real-world AI challenges (the goal being to work with unseen data) and b) you canot train your network on the marking samples to gain an unfair advantage (e.g. deliberately making your network overfit to the marking samples).

The argument to '-d' will be the directory we have asked you to place you network to be marked in.  **Make sure** this works, when a suitable mark.npy in the script directory exists, with your team's submission directory:
```
sh mark.sh -d /rds/projects/2018/thompssj-bear-chal18-0X/chal5
````

Most accurate on our marking datasets wins.

Good luck!

