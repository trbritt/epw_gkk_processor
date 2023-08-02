# epw_gkk_processor

Hi there. Welcome.

You've arrived to the repository for a little post-processing utility. My *ab-initio* calculations using the `EPW` package of the `QuantumESPRESSO` suite has lead me to an interesting crossroads. 

## Goal: be able to synthesize the electron-phonon coupling matrix elements that are printed as human-readable text into a usable format

Unfortunately, the `EPW` programme as is will return $|g|$ (meV) for a given phonon and electron momenta grids, specified electronic bands, and all phonon modes, but only in human readable format when
running the programme using the prtgkk flag. This is extremely unfortunate, as this produces text files 100s of GBs (if not terrabytes) in size for grids with density sufficient
for reasonable physical inference. 

## Solution: create high performance text processor to output the desired data in a usable format


To this aim, I created this program which performs the following:

- Regex searches the entire text file to be able to determine the blocks of relevant data
     - The programme outputs data by the $q$ value, so the 'block' referred to here is the block corresponding to a given value of $q$
- Distributes the blocks of data to serial archives for the number of MPI processes currently executing the programme
     - Should the user rerun the anaylsis (post regex processing) with a different number of cores, the programme will redistribute the archives to the current number of processes
- Each MPI process then begins processing the blocks of data for which it is deemed responsible. This entails:
     - Regexing the block to determine the regions corresponding to data for a given $k$ value
     - Processing each block to identify the data
     - Writing the data per block into a hyperslab of an HDF5 file in parallel
Effecient regexing is performed by help of the optimized regex engine of the Boost library. Efficient parallelization is provided by both OpenMPI and OpenMP parallelization: MPI processes have distributed among them different blocks of q-data, while each process spawns OpenMP threads to process a given k-block of data at a time.

## Compilation

To compile this program, simply run `cmake` in the source directory. CMake will attempt to identify the needed libraries (which are MPI, HDF5, and Boost). Should the libraries be provided, simply run `make` to perform the actual compilation.

As of 02 August 2023, the switch to toggle between processing from the direct text output of `EPW` and to start processing post regexing is hard coded as a precompiler directive, and therefore recompilation must be done before switching the programme between
its two primary functions.

## Result

The result of the programme is an HDF5 file with the following hierarchy:

<prefix>.hdf5
├── grids
│   ├── kpts
│   └── qpts
└── epmat
    └── dataset

The `epmat` dataset is the entire 5D array of data returned by `EPW`. 
