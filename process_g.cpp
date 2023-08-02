#include "process_g.h"
#include <filesystem>

#define FAIL -1
#define ARCHIVE_BLOCK_PREFIX "block_p"
#define ARCHIVE_QGRID_NAME "qgrid.ser"
#define PROCESS_FROM_SCRATCH 0

namespace mpi = boost::mpi;


int main(int argc, char *argv[])
{
    int rank, nproc;

    mpi::environment env;
    mpi::communicator world;

    rank = world.rank();
    nproc = world.size();

    const int NQPTS = 256;
    const int NKPTS = 2048;
    const int NBNDS = 13;
    const int NMODES = 24;
    const int QMESH[3] = {1, 16, 16};
    const int KMESH[3] = {8, 16, 16};
    const std::streamsize CHUNK_SIZE = std::pow(2, 35) - 1;
    char lVersionRT[MPI_MAX_LIBRARY_VERSION_STRING] = {0};

    int lLongRT = 0;
    MPI_Get_library_version(lVersionRT, &lLongRT);
    unsigned int thread_qty;
    try {
        thread_qty = std::max(atoi(std::getenv("OMP_NUM_THREADS")), 1);
    } catch (...) { //avoid if the user does not have OMP_NUM_THREADS defined in the env
        thread_qty = 1;
    }

    omp_set_num_threads(thread_qty);

    if (rank == 0)
    {
        // Print header
        // Get the current time
        std::chrono::system_clock::time_point now = std::chrono::system_clock::now();

        // Convert the time to a time_t representation (seconds since epoch)
        std::time_t now_time_t = std::chrono::system_clock::to_time_t(now);

        // Convert time_t to a local time representation (struct tm)
        std::tm local_time = *std::localtime(&now_time_t);

        std::string header = "";
        header += BColors::HEADER + "###################################################################\n";
        header += "# EPW2PY v0.1\n";
        if (nproc == 1)
        {
            header += "# Running in SERIAL MODE\n";
        }
        else
        {
            header += "# Running in PARALLEL MODE on " + std::to_string(nproc) + " processes\n";
        }
        if (omp_get_max_threads() <= 1){
            header += "# Running with no OpenMP parallelization\n";
        } else {
            header += "# Running with OpenMP parallelization on " + std::to_string(omp_get_max_threads()) + " threads\n";
        }
        header += std::string("# Built on ") + lVersionRT + "\n"; // operator+ goes left right, so const char+char is not defined, but str+char is, so explicitly cast first part
        header += "# Built on HDF5 " + std::to_string(H5_VERS_MAJOR) + "." + std::to_string(H5_VERS_MINOR) + "." + std::to_string(H5_VERS_RELEASE) + "\n";
        header += "#\n#\tqmesh: " + std::to_string(QMESH[0]) + " " + std::to_string(QMESH[1]) + " " + std::to_string(QMESH[2]) + "\n";
        header += "#\tkmesh: " + std::to_string(KMESH[0]) + " " + std::to_string(KMESH[1]) + " " + std::to_string(KMESH[2]) + "\n";
        header += "#\tnbnds: " + std::to_string(NBNDS) + "\n";
        header += "#\tnmodes: " + std::to_string(NMODES) + "\n";
        header += "###################################################################" + BColors::ENDC + "\n";
        std::cout << word_wrap(header, 71) << std::endl;
        std::cout << "# Starting on " << std::put_time(&local_time, "%Y-%m-%d %H:%M:%S") << std::endl;
    }

    // Create HDF5 file
    // declare the file ID
    std::string filename = "SnSe_pnma.epmat.h5";
    /* setup file access template with parallel IO access. */
    hid_t acc_tpl = H5Pcreate(H5P_FILE_ACCESS);
    assert(acc_tpl != FAIL);


    /* set Parallel access with communicator */
    herr_t ret = H5Pset_fapl_mpio(acc_tpl, MPI_COMM_WORLD, MPI_INFO_NULL);
    assert(acc_tpl != FAIL);

    /* create the file collectively */
    hid_t file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, acc_tpl);
    assert(file_id != FAIL);

    /* Release file-access template */
    ret = H5Pclose(acc_tpl);
    assert(ret != FAIL);

    /* setup dimensionality object */
    hsize_t qpt_dims[2] = {NQPTS, 3};
    hid_t qpt_dataspace_id = H5Screate_simple(2, qpt_dims, NULL);
    assert(qpt_dataspace_id != FAIL); //sid in template

    hsize_t kpt_dims[2] = {NKPTS, 3};
    hid_t kpt_dataspace_id = H5Screate_simple(2, kpt_dims, NULL);
    assert(kpt_dataspace_id != FAIL); //sid in template

    hsize_t epmat_dims[5] = {NQPTS, NKPTS, NBNDS, NBNDS, NMODES};
    hid_t epmat_dataspace_id = H5Screate_simple(5, epmat_dims, NULL);
    assert(epmat_dataspace_id != FAIL); //sid in template

    /* create a group collectively */
    hid_t grids_group_id = H5Gcreate(file_id, "/grids", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    assert(grids_group_id != FAIL);

    /* create a dataset collectively */
    hid_t qpt_dataset_id = H5Dcreate2(grids_group_id, "qpts", H5T_NATIVE_DOUBLE, qpt_dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    assert(qpt_dataset_id != FAIL);
    hid_t kpt_dataset_id = H5Dcreate2(grids_group_id, "kpts", H5T_NATIVE_DOUBLE, kpt_dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    assert(kpt_dataset_id != FAIL);
    hid_t epmat_dataset_id = H5Dcreate2(file_id, "epmat", H5T_NATIVE_DOUBLE, epmat_dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    assert(epmat_dataset_id != FAIL);

    // Calculate bounds for each process
    std::pair<int, int> local_bounds = fqbounds(NQPTS, rank, nproc);
    int lower_bnd = local_bounds.first;
    int upper_bnd = local_bounds.second;

    std::vector<std::pair<int, int>> bounds(nproc);
    for (int j = 0; j < nproc; j++)
    {
        bounds[j] = fqbounds(NQPTS, j, nproc);
    }
    std::vector<int> elementCounts(nproc, 0);
    for (int j = 0; j < nproc; j++)
    {
        elementCounts[j] = bounds[j].second - bounds[j].first + 1;
    }
    std::vector<int> displacements(nproc, 0);
    for (int i = 1; i < nproc; ++i)
    {
        displacements[i] = displacements[i - 1] + elementCounts[i - 1];
    }

    std::vector<double> unfolded_qpts(NQPTS * 3); //this needs to be contiguous in memory

    std::vector<Timer> timers(3);
    timers[0].name = "load";
    timers[1].name = "regex";
    timers[2].name = "idq processing";

    if (rank == 0)
    {
        std::cout << "Elements to be handled per proc: " << elementCounts << std::endl;

        std::cout << BColors::OKBLUE << "===================================================================" << std::endl;
        std::cout << "All processes see dataset. Beginning processing..." << std::endl;
        std::cout << "===================================================================" << BColors::ENDC << std::endl;
        const boost::regex q_point_pattern(
                R"(iq\s*=\s*(\d+)\s*coord.:\s*(\d+\.\d+)\s*(\d+\.\d+)\s*(\d+\.\d+))");
        if (PROCESS_FROM_SCRATCH == 1){
            std::ifstream file("SnSe_pnma.linewidth.epw.out");
            if (!file.is_open())
            {
                std::cerr << "Error opening file." << std::endl;
                return 1;
            }
            // below is for reading in the file directly with no progress bar
            // std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
            // file.close();
            // Get the file size to calculate the total bytes read
            timers[0].start();
            file.seekg(0, std::ios::end);
            std::streampos fileSize = file.tellg();
            file.seekg(0, std::ios::beg);

            // Allocate memory for the content dynamically
            char *content = new char[fileSize + 1];
            content[fileSize] = '\0'; // Null-terminate the content buffer

            // Calculate the number of chunks needed
            std::streamsize numChunks = (fileSize + CHUNK_SIZE - 1) / CHUNK_SIZE;

            std::streamsize bytesRead = 0;
            int prevProgress = -1;
            for (std::streamsize i = 0; i < numChunks; ++i)
            {
                std::streamsize chunkSize = (i == numChunks - 1) ? static_cast<std::streamsize>(fileSize - bytesRead) : CHUNK_SIZE;
                file.read(content + bytesRead, chunkSize);
                std::streamsize bytesReadInChunk = file.gcount();

                bytesRead += bytesReadInChunk;
                assertm(bytesReadInChunk == chunkSize, "Didn't seem to read in the right amount of data");

                updateProgressBar(static_cast<float>(bytesRead) / fileSize * 100, 100, 64);
                // std::cout << bytesReadInChunk << "/" << fileSize << " - " << CHUNK_SIZE << std::endl;
            }
            std::cout << std::endl;
            file.close();
            // Assert that we've read in all of the data from the file!
            assertm(bytesRead == fileSize, "Didn't seem to read in the entire file");
            timers[0].stop();
            //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            // Begin processing the regexes ...
            //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            timers[1].start();

            std::cout << BColors::OKBLUE << "===================================================================" << std::endl;
            std::cout << "Regex finding qpt blocks ..." << std::endl;
            std::cout << "===================================================================" << BColors::ENDC << std::endl;
            std::cout << BColors::OKCYAN << "-------------------------------------------------------------------" << std::endl;
            std::cout << "Finding qpt data blocks ..." << std::endl;
            std::cout << "-------------------------------------------------------------------" << BColors::ENDC << std::endl;
            // Now q_point_data_matches contains the list of valid matches.
            boost::cregex_token_iterator iter_block(content, content + bytesRead, q_point_pattern, {-1, 1});
            boost::cregex_token_iterator end_block;

            int idm = -1;
            int current_responsible_process = 0;
            int counter=0;
            std::vector<std::string> localData;
            for (; iter_block != end_block; ++iter_block)
            {
                const std::string match = iter_block->str();
                if (!match.empty() && match.size() > 10000)
                {
                    // this means we have a successful match...
                    idm++;
                    counter++;                    
                    localData.push_back(match);

                    // the check below sees if we are in the region given by the current responsible process, if
                    // we are not, then we need to send the data accumulated to the local responsible proc
                    // and increment the current responsible process
                    if (idm == bounds[current_responsible_process].second)
                    {
                        std::cout << "Found for " << current_responsible_process << " " << localData.size() << " elements, expected " << elementCounts[current_responsible_process] << std::endl;
                        assertm(localData.size() == elementCounts[current_responsible_process], std::string(std::string("Did not find all blocks that proc ") + std::to_string(current_responsible_process) + std::string(" was supposed to find.")));
                        // Serialize the data into a continuous buffer
                        {
                            std::ofstream ofs(std::string(ARCHIVE_BLOCK_PREFIX) + std::to_string(current_responsible_process)+".ser");
                            boost::archive::text_oarchive oa(ofs);
                            oa & localData;
                        }
                        localData.clear();
                        current_responsible_process++;
                        counter=0;
                    }
                    // the above check does the communication of the buffer when it is needed for the
                    // newest index idm. But in either case, we take the match and add it to the running buffer
                    // it's just that it may be freshly depleted from the above sending
                    
                    updateProgressBar(static_cast<float>(idm) / NQPTS * 100, 100, 64);
                }
            }
            std::cout << std::endl;
            assertm(idm + 1 == NQPTS, "Trouble loading some qpt blocks successfully.");
            
            std::cout << BColors::OKCYAN << "*******************************************************************" << std::endl;
            std::cout << " Successfully saved qpt data blocks to serial archive ..." << std::endl;
            std::cout << "*******************************************************************" << BColors::ENDC << std::endl;
            std::cout << BColors::OKCYAN << "-------------------------------------------------------------------" << std::endl;
            std::cout << "Finding qpt values ..." << std::endl;
            std::cout << "-------------------------------------------------------------------" << BColors::ENDC << std::endl;

            // we need to now identify the qpts themselves!
            double *dataptr = &unfolded_qpts[0];
            idm = -1;
            boost::cregex_iterator iter(content, content + bytesRead, q_point_pattern);
            boost::cregex_iterator end;

            for (boost::cregex_iterator i = iter; i != end; ++i)
            {
                idm++;
                boost::cmatch match = *i;
                // in theory, match.str() will return the entire match
                // and then match[1] returns first subgroup (aka iq),
                // match[2] returns second, which is coord 1, etc...

                *dataptr++ = std::stod(match[2].str());
                *dataptr++ = std::stod(match[3].str());
                *dataptr++ = std::stod(match[4].str());

                updateProgressBar(static_cast<float>(idm) / NQPTS * 100, 100, 64);
            }
            
            std::cout << std::endl;
            assertm(idm + 1 == NQPTS, "Trouble loading some qpt values successfully.");
            {
                std::ofstream ofs(std::string(ARCHIVE_QGRID_NAME));
                boost::archive::text_oarchive oa(ofs);
                oa & unfolded_qpts;
            }
            std::cout << BColors::OKCYAN << "*******************************************************************" << std::endl;
            std::cout << " Successfully saved qpt data to serial archive ..." << std::endl;
            std::cout << "*******************************************************************" << BColors::ENDC << std::endl;
            // release original data array
            delete[] content;
            timers[1].stop();

        } else { //the archive exists so load it!
            std::cout << BColors::OKCYAN << "*******************************************************************" << std::endl;
            std::cout << "Loading qpt data from serial archive..." << std::endl;
            std::cout << "*******************************************************************" << BColors::ENDC << std::endl;
            timers[0].start();
            //the above processing from scratch will write archives for the given number
            //of procs it sees at that time, which may be different than what it sees here.
            //therefore, if the number of procs is different, we need to combiner
            //all archives and redistribute
            //first, verify current number of archives
            int n_saved_archives=0;
            for (const auto& entry : std::filesystem::directory_iterator(".")) {
                if (entry.path().filename().string().find("block") != std::string::npos) {
                    n_saved_archives++;
                }
            }
            if (n_saved_archives != nproc){
                std::cout << BColors::OKCYAN << "*******************************************************************" << std::endl;
                std::cout << "Redistributing data from " << n_saved_archives <<" archives to " << nproc <<" current processes..." << std::endl;
                std::cout << "*******************************************************************" << BColors::ENDC << std::endl;
                //combine all existing archives on the root process
                std::vector<std::string> combined_archives;
                for (int i=0; i<n_saved_archives; i++){
                    std::vector<std::string> tmp;
                    std::string archive_name = std::string(ARCHIVE_BLOCK_PREFIX) + std::to_string(i)+".ser";
                    std::ifstream ifs(archive_name);
                    boost::archive::text_iarchive ia(ifs);
                    ia & tmp;
                    combined_archives.insert(combined_archives.end(), tmp.begin(), tmp.end());
                    updateProgressBar(static_cast<float>(i) / n_saved_archives * 100, 100, 64);
                }
                std::cout << std::endl;

                assertm(combined_archives.size() == NQPTS, "Did not load expected number of blocks from disk.")
                //delete current archives
                for (const auto& entry : std::filesystem::directory_iterator(".")) {
                    if (entry.path().filename().string().find("block") != std::string::npos) {
                        std::filesystem::remove(entry.path());
                    }
                }
                //now redistribute to disk
                for (int i=0; i<nproc; i++){
                    std::pair<int, int> correct_bounds = fqbounds(NQPTS, i, nproc);
                    std::vector<std::string>::const_iterator first = combined_archives.begin() + correct_bounds.first;
                    std::vector<std::string>::const_iterator last = combined_archives.begin() + correct_bounds.second + 1;
                    std::vector<std::string> localData(first, last);
                    std::ofstream ofs(std::string(ARCHIVE_BLOCK_PREFIX) + std::to_string(i)+".ser");
                    boost::archive::text_oarchive oa(ofs);
                    oa & localData;
                    updateProgressBar(static_cast<float>(i) / nproc * 100, 100, 64);
                }
                std::cout << std::endl;
            }
            
            std::ifstream ifs(ARCHIVE_QGRID_NAME);
            boost::archive::text_iarchive ia(ifs);
            ia & unfolded_qpts;
            
            assertm(unfolded_qpts.size() == static_cast<int>(NQPTS*3), "Did not load expected number of qpts from archive");

            timers[0].stop();
            timers[1].start();
            timers[1].stop();
        }
        

        /* create a file dataspace independently */
        hid_t file_dataspace = H5Dget_space(qpt_dataset_id);
        assert(file_dataspace != FAIL);

        /* set up dimensions of the slab this process accesses */
        hsize_t start[2];                      /* for hyperslab setting */
        hsize_t count[2], stride[2];           /* for hyperslab setting */
        start[0]  = 0;
        start[1]  = 0;
        count[0]  = NQPTS;
        count[1]  = 3;
        stride[0] = 1;
        stride[1] = 1;

        /* create a hyperslab independently */
        ret = H5Sselect_hyperslab(file_dataspace, H5S_SELECT_SET, start, stride, count, NULL);
        assert(ret != FAIL);

        /* create a memory dataspace independently */
        hid_t mem_dataspace = H5Screate_simple(2, count, NULL);
        assert(mem_dataspace != FAIL);

        /* write data independently */
        ret = H5Dwrite(qpt_dataset_id, H5T_NATIVE_DOUBLE, mem_dataspace, file_dataspace, H5P_DEFAULT, unfolded_qpts.data());
        assert(ret != FAIL);

        /* release dataspace ID */
        H5Sclose(file_dataspace);
        H5Sclose(mem_dataspace);

        // All necessary data is now stored in q_point_data_matches and unfolded_qpts
    }
    world.barrier(); //root process needs to finish computing from scratch, redistributing, or other

    /* close dataset collectively */
    H5Dclose(qpt_dataset_id);
    /* release all IDs created */
    H5Sclose(qpt_dataspace_id);

    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    // each proc loads its archive
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    std::vector<std::string> localData;
    {
        std::ifstream ifs(std::string(ARCHIVE_BLOCK_PREFIX) + std::to_string(rank)+".ser");
        boost::archive::text_iarchive ia(ifs);
        ia & localData;
    }
    // std::cout << "Rank " << rank << "found " << localData.size() << " elements, expected" << elementCounts[rank] << std::endl;
    world.barrier();
    assertm(localData.size() == elementCounts[rank], std::string("Did not load correct amount of blocks on proc#" + std::to_string(rank)));
    
    world.barrier();
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    // get ready to process each block
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    // Define the regex pattern for K_POINT_PATTERN
    const boost::regex k_point_pattern(R"(ik\s*=\s*(\d+)\s*coord.:\s*(\d+\.\d+)\s*(\d+\.\d+)\s*(\d+\.\d+))");
    if (rank == 0)
    {
        timers[2].start();
        std::cout << BColors::OKBLUE << "===================================================================" << std::endl;
        std::cout << "Regex finding kpt blocks ..." << std::endl;
        std::cout << "===================================================================" << BColors::ENDC << std::endl;
    }
    std::vector<int> is_q_processed(localData.size(), 0);
    //slight issue, due to the gather operation at the end of the loop,
    //because some procs iterate for less items than others, they will
    //exit the loop, and therefore not call the collective gather
    //operation on the final iteration of the loop, so the root
    //process hangs indefinitely ...
    //Fix: ensure all processes iterate for the same amount of items
    //just enforce that for the left over iterations, they don't 
    //do any processing, just that they call the final gather statements
    std::vector<int>::iterator max_n_iterations;
    max_n_iterations = std::max_element(elementCounts.begin(), elementCounts.end());
    // if (rank==0){std::cout<<"Each proc will iterate " << *max_n_iterations << " times" << std::endl;}
    for (int idq = 0; idq < *max_n_iterations; idq++) //each process iterates the maximum amount
    {
        if (idq < localData.size()){
            // and then we also need to get the actual kpoints!
            std::vector<double> unfolded_kpts(NKPTS * 3); //this needs to be contiguous in memory
            double *dataptr = &unfolded_kpts[0];
            boost::sregex_iterator iter_kpt(localData[idq].begin(), localData[idq].end(), k_point_pattern);
            boost::sregex_iterator end_kpt;

            int idm = -1;
            for (boost::sregex_iterator i = iter_kpt; i != end_kpt; ++i)
            {
                idm++;
                boost::smatch match = *i;
                *dataptr++ = std::stod(match[2].str());
                *dataptr++ = std::stod(match[3].str());
                *dataptr++ = std::stod(match[4].str());
            }

            assertm(idm + 1 == NKPTS, "Trouble loading some kpt values successfully.");
            if (rank==0){ //write dataset only once
                /* create a file dataspace independently */
                hid_t file_dataspace = H5Dget_space(kpt_dataset_id);
                assert(file_dataspace != FAIL);

                /* set up dimensions of the slab this process accesses */
                hsize_t start[2];                      /* for hyperslab setting */
                hsize_t count[2], stride[2];           /* for hyperslab setting */
                start[0]  = 0;
                start[1]  = 0;
                count[0]  = NKPTS;
                count[1]  = 3;
                stride[0] = 1;
                stride[1] = 1;

                /* create a hyperslab independently */
                ret = H5Sselect_hyperslab(file_dataspace, H5S_SELECT_SET, start, stride, count, NULL);
                assert(ret != FAIL);

                /* create a memory dataspace independently */
                hid_t mem_dataspace = H5Screate_simple(2, count, NULL);
                assert(mem_dataspace != FAIL);

                /* write data independently */
                ret = H5Dwrite(kpt_dataset_id, H5T_NATIVE_DOUBLE, mem_dataspace, file_dataspace, H5P_DEFAULT, unfolded_kpts.data());
                assert(ret != FAIL);

                /* release dataspace ID */
                H5Sclose(file_dataspace);
                H5Sclose(mem_dataspace);
            }

            // Now that we finally have a qpoint block, we need
            // to examine it to identify all regions of kpt data
            std::vector<std::string> k_point_data_matches;
            boost::sregex_token_iterator iter_kmatches(
                localData[idq].begin(), 
                localData[idq].end(), 
                k_point_pattern, 
                -1
            );
            boost::sregex_token_iterator end_kmatches;
            int idk = -1;
            if (rank == 0)
            {
                std::cout << "Finding blocks ..." << std::endl;
            }
            for (boost::sregex_token_iterator i = iter_kmatches; i != end_kmatches; ++i)
            {
                std::string match = (*i).str();
                if (!match.empty() && match.find("\n") != std::string::npos && match.size() > 1000)
                {
                    k_point_data_matches.push_back(match);
                    idk++;
                    if (rank==0) {updateProgressBar(static_cast<float>(idk) / NKPTS * 100, 100, 64);}
                }
            }
            if (rank==0) {std::cout << std::endl;}

            assertm(idk == idm, "Found different number of kpts than kpt data blocks!")
            for (idk = 0; idk < k_point_data_matches.size(); ++idk){
                size_t header_pos = k_point_data_matches[idk].find("\n");
                size_t footer_pos = k_point_data_matches[idk].rfind("\n");
                k_point_data_matches[idk] = k_point_data_matches[idk].substr(header_pos + 1, footer_pos - header_pos - 1);
            }                // Remove header and footer from the data


            //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            // now actually process the kblocks to get the resulting 
            // data. We're at fixed q right now, so let's make an
            // array that is NKPTS, NBNDS, NBNDS, NMODES large. We 
            // iterate over k_point_data_matches to fill the first
            // dimension and each item of k_point_data_matches
            // is what we send to process_k_block which returns an
            // array of size NBNDS, NBNDS, NMODES to be passed to the
            // global array here...
            //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            // When you assign a vector of vectors of vectors to 
            // the first index, you are effectively updating an
            // entire 3D "slice" of the 4D array. In Fortran storage 
            // order, the elements of each column (second dimension)
            // are stored together, so updating a full 2D slice 
            // (3rd and 4th dimensions) at once can be more
            // cache-friendly and efficient.

            array4d kblock_array(
                boost::extents[NKPTS][NBNDS][NBNDS][NMODES], 
                boost::fortran_storage_order()
            );


            /* Below is for zero OpenMP parallelization, then for OpenMP parallelization*/
            std::vector<int> is_k_processed(k_point_data_matches.size(), 0);
            if (omp_get_max_threads() <= 1){
                if (rank == 0)
                {
                    std::cout << "Processing " << static_cast<int>(k_point_data_matches.size()*nproc) << " blocks on no OpenMP threads..." << std::endl;
                }
                index4d i4d = -1;
                while (!k_point_data_matches.empty())
                {
                    i4d++;
                    kblock_array[i4d] = process_k_block(
                        k_point_data_matches[i4d], 
                        NBNDS, NBNDS, NMODES
                    );
                    is_k_processed[i4d] = 1;
                    if (rank==0) {updateProgressBar(static_cast<float>(i4d) / NKPTS * 100, 100, 64);}
                }
            } else {
                if (rank == 0)
                {
                    std::cout << "Processing " << static_cast<int>(k_point_data_matches.size()*nproc) << " blocks on " <<  omp_get_max_threads() << " OpenMP threads..." << std::endl;
                }
                #pragma omp parallel for num_threads(omp_get_max_threads())
                for (index4d i4d = 0; i4d < k_point_data_matches.size(); i4d ++){
                    std::string localData;
                    localData = k_point_data_matches[i4d];
                    if (!localData.empty()) {
                        kblock_array[i4d] = process_k_block(localData, NBNDS, NBNDS, NMODES);
                        is_k_processed[i4d] = 1;
                    }                
                    // Use the master thread (thread 0) to update the progress bar
                    if (omp_get_thread_num() == 0 && rank==0) {
                        int result = std::reduce(is_k_processed.begin(), is_k_processed.end());
                        updateProgressBar(static_cast<float>(result) / NKPTS * 100, 100, 64);
                    }
                }
            }//end processing k blocks with(out) omp
            // std::cout << is_processed << std::endl;
            assertm(std::reduce(is_k_processed.begin(), is_k_processed.end()) == NKPTS, "Did not process all blocks correctly.")
            if (rank==0){std::cout << std::endl;}

            std::vector<double> flattened_kblock_array(NKPTS*NBNDS*NBNDS*NMODES);
            dataptr = &flattened_kblock_array[0];
            for (int i=0; i<NKPTS; i++){
                for (int j=0; j<NBNDS; j++){
                    for (int k=0; k<NBNDS; k++){
                        for (int l=0; l<NMODES; l++){
                            *dataptr++ = kblock_array[i][j][k][l];
                        }
                    }
                }
            }//end flattening kblock_array

            /* create a file dataspace independently */
            hid_t file_dataspace = H5Dget_space(epmat_dataset_id);
            assert(file_dataspace != FAIL);

            /* set up dimensions of the slab this process accesses */
            long unsigned int current_index_in_slab = lower_bnd+idq;
            // for (int r=0; r<nproc; r++){
            //     world.barrier();
            //     if (r == rank) {
            //         std::cout << "Proc "<< rank << " writes to dset " << current_index_in_slab << std::endl;
            //     }
            // }
            hsize_t start[5] = {current_index_in_slab, 0, 0, 0, 0}; // Start index for this process
            hsize_t count[5] = {1, NKPTS, NBNDS, NBNDS, NMODES}; // Number of elements for this process

            /* create a hyperslab independently */
            ret = H5Sselect_hyperslab(file_dataspace, H5S_SELECT_SET, start, NULL, count, NULL);
            assert(ret != FAIL);

            /* create a memory dataspace independently */
            hid_t mem_dataspace = H5Screate_simple(5, count, NULL);
            assert(mem_dataspace != FAIL);

            /* write data independently */
            ret = H5Dwrite(epmat_dataset_id, H5T_NATIVE_DOUBLE, mem_dataspace, file_dataspace, H5P_DEFAULT, flattened_kblock_array.data());
            assert(ret != FAIL);
            
            /* release dataspace ID */
            H5Sclose(file_dataspace);
            H5Sclose(mem_dataspace);

            //report to the master process on progress (sounds intimidating right? almost like your boss???)
            is_q_processed[idq] = 1;
        }//if idq < localData.size(), enforces each proc iterates the same amount
        int processed_q_so_far = std::reduce(is_q_processed.begin(), is_q_processed.end());
        world.barrier();
        if (rank == 0)
        {
            std::vector<int> processed_per_proc;
            gather(world, processed_q_so_far, processed_per_proc, 0);
            std::cout << std::endl << BColors::OKCYAN << "-------------------------------------------------------------------" << std::endl;
            std::cout << "Finished processing the following per proc: \n";
            for (int r=0; r<nproc; r++){
                std::cout << processed_per_proc[r] << "/" << elementCounts[r] << " ";
            }
            std::cout << std::endl << "-------------------------------------------------------------------" << BColors::ENDC << std::endl;
        } else {
            gather(world, processed_q_so_far, 0);
        }
    }
    assertm(std::reduce(is_q_processed.begin(), is_q_processed.end())==localData.size(), "Did not process all qpt blocks correctly.");
    //assert all procs have finished, more for debugging
    for (int r = 0; r < nproc; ++r) {
        world.barrier(); //order of processes in the output will remain the same because of the barrier!
        if (r == rank) {
            std::cout << "Rank " << r << " finished processing ..." << '\n';
        }
    }
    // free resources and finalize MPI

    /* close dataset collectively */
    ret = H5Dclose(epmat_dataset_id);
    assert(ret != FAIL);
    if(rank==0){std::cout << "Closed epmat dataset, ";}

    /* release all IDs created */
    ret = H5Sclose(epmat_dataspace_id);
    assert(ret != FAIL);
    if(rank==0){std::cout << "closed epmat dataspace, ";}

    ret = H5Dclose(kpt_dataset_id);
    assert(ret != FAIL);
    if(rank==0){std::cout << "closed kpt dataset, ";}

    ret = H5Sclose(kpt_dataspace_id);
    assert(ret != FAIL);
    if(rank==0){std::cout << "closed kpt dataspace, ";}

    ret = H5Gclose(grids_group_id);
    assert(ret != FAIL);
    if(rank==0){std::cout << "closed grids group, ";}

    H5Fclose(file_id);
    if(rank==0){std::cout << "closed hdf5 file." << std::endl;}

    if (rank == 0)
    {
        timers[2].stop();
        // print timing information
        std::cout << std::endl << "-------------------------------------------------------------------" << std::endl;
        for (int i = 0; i < timers.size(); i++)
        {
            timers[i].print();
        }
        std::cout << "-------------------------------------------------------------------" << std::endl;
        // Get the current time
        std::chrono::system_clock::time_point now = std::chrono::system_clock::now();

        // Convert the time to a time_t representation (seconds since epoch)
        std::time_t now_time_t = std::chrono::system_clock::to_time_t(now);

        // Convert time_t to a local time representation (struct tm)
        std::tm local_time = *std::localtime(&now_time_t);

        std::cout << BColors::OKGREEN << "###################################################################" << std::endl;
        std::cout << "Cleanly exiting; ended on " << std::put_time(&local_time, "%Y-%m-%d %H:%M:%S") << " ... Goodbye." << std::endl;
        std::cout << "###################################################################" << BColors::ENDC << std::endl;
    }
    MPI_Finalize();
    return 0;
}
