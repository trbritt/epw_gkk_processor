#include <iostream>
#include <iomanip>
#include <fstream>

#include <chrono>
#include <ctime>
#include <cctype>
#include <cmath>

#include <algorithm>
#include <iterator>

#include <vector>
#include <string>

#include <boost/regex.hpp>
#include <boost/multi_array.hpp>
#include <boost/algorithm/string/classification.hpp> // Include boost::for is_any_of
#include <boost/algorithm/string/split.hpp> // Include for boost::split
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/string.hpp>

#include <cassert>

#include <boost/mpi.hpp>

#include <hdf5.h>
#include <hdf5_hl.h>

#include <omp.h>

typedef boost::multi_array<double, 3> array3d;
typedef array3d::index index3d;

typedef boost::multi_array<double, 4> array4d;
typedef array4d::index index4d;


#define assertm(exp, msg) assert(((void)msg, exp));


// Function to process a single (q,k) block of data
array3d process_k_block(const std::string& db, int maxibnd, int maxjbnd, int maxkbnd) {

    std::vector<std::string> datablock;
    boost::split(datablock, db, boost::is_any_of("\n"), boost::token_compress_on);
    size_t minLength = 90;
    // Use erase-remove idiom to remove strings shorter than minLength
    datablock.erase(
        std::remove_if(
            datablock.begin(), 
            datablock.end(),
            [minLength](const std::string& str) { 
                return str.length() < minLength; 
            }
        ),
        datablock.end()
    );
    // Regex pattern to extract numerical values from each row
    boost::regex num_pattern(R"-(-?\d+(?:\.\d+)?(?:E[+-]?\d+)?)-");

    array3d retval(boost::extents[maxibnd][maxjbnd][maxkbnd]);

    assertm(datablock.size() == static_cast<int>(maxibnd*maxjbnd*maxkbnd), "Found different amount of points that expected");
    for (int id=0; id<datablock.size(); id++) {
        boost::sregex_iterator iter(
            datablock[id].begin(), 
            datablock[id].end(), 
            num_pattern
        );
        boost::sregex_iterator end;
        std::vector<double> match_values;
        for (boost::sregex_iterator i = iter; i != end; ++i) {
            boost::smatch match = *i;
            match_values.push_back(std::stod(match.str()));
        }
        index3d ibnd = static_cast<int>(match_values[0]) - 1;
        index3d jbnd = static_cast<int>(match_values[1]) - 1;
        index3d nmode = static_cast<int>(match_values[2]) - 1;
        retval[ibnd][jbnd][nmode] = match_values[6];
    }
    return retval; 
}


//pretty print vector
template<typename T>
std::ostream & operator<<(std::ostream & os, std::vector<T> vec)
{
    os<<"{";
    if(vec.size()!=0)
    {
        std::copy(vec.begin(), vec.end()-1, std::ostream_iterator<T>(os, " "));
        os<<vec.back();
    }
    os<<"}";
    return os;
}

//pop front and return
template<typename T>
T pop_front(std::vector<T>& vec)
{
    assert(!vec.empty());
    T tmp = vec.front();
    vec.erase(vec.begin());
    return tmp;
}

std::string word_wrap(std::string text, unsigned per_line)
{
    unsigned line_begin = 0;

    while (line_begin < text.size())
    {
        const unsigned ideal_end = line_begin + per_line ;
        unsigned line_end = ideal_end < text.size() ? ideal_end : text.size()-1;

        if (line_end == text.size() - 1)
            ++line_end;
        else if (std::isspace(text[line_end]))
        {
            text[line_end] = '\n';
            ++line_end;
        }
        else    // backtrack
        {
            unsigned end = line_end;
            while ( end > line_begin && !std::isspace(text[end]))
                --end;

            if (end != line_begin)                  
            {                                       
                line_end = end;                     
                text[line_end++] = '\n';            
            }                                       
            else                                    
                text.insert(line_end++, 1, '\n');
        }

        line_begin = line_end;
    }

    return text;
}
class BColors {
public:
    static const std::string HEADER;
    static const std::string OKBLUE;
    static const std::string OKCYAN;
    static const std::string OKGREEN;
    static const std::string WARNING;
    static const std::string FAIL;
    static const std::string ENDC;
    static const std::string BOLD;
    static const std::string UNDERLINE;
};

const std::string BColors::HEADER = "\033[95m";
const std::string BColors::OKBLUE = "\033[94m";
const std::string BColors::OKCYAN = "\033[96m";
const std::string BColors::OKGREEN = "\033[92m";
const std::string BColors::WARNING = "\033[93m";
const std::string BColors::FAIL = "\033[91m";
const std::string BColors::ENDC = "\033[0m";
const std::string BColors::BOLD = "\033[1m";
const std::string BColors::UNDERLINE = "\033[4m";


// Helper function to calculate the bounds for each process
std::pair<int, int> fqbounds(int nqtot, int rank, int nproc) {
    int lower_bnd, upper_bnd;
    if (nproc == 1) {
        lower_bnd = 0;
        upper_bnd = nqtot-1;
    } else {
        int nkl = nqtot / nproc;
        int nkr = nqtot - nkl * nproc;
        if (rank < nkr) {
            nkl = nkl + 1;
        }
        lower_bnd = rank * nkl + 1;
        if (rank >= nkr) {
            lower_bnd = rank * nkl + 1 + nkr;
        }
        upper_bnd = lower_bnd + nkl - 1;
    }
    return std::make_pair(lower_bnd - 1, upper_bnd - 1);
}

void updateProgressBar(float current, int total, int barWidth = 70) {
    float progress = static_cast<float>(current) / total;
    int pos = static_cast<int>(barWidth * progress);

    std::cout << "[";
    for (int i = 0; i < barWidth; ++i) {
        if (i + 7 < pos) {std::cout << "=";}
        else if (i + 7 == pos) {std::cout << ">";}
        else {std::cout << " ";}
    }
    std::cout << "] " << std::setprecision(5) << progress * 100.0 << "%\r" << std::flush;

    if (static_cast<int>(current) == total) std::cout << std::endl;
}

class Timer {
    public:
        typedef std::chrono::high_resolution_clock Clock;
        std::string name;
        void start(){ 
            epoch = Clock::now(); 
        }
        void stop() { 
            telapsed = Clock::now() - epoch;
        }
        void print() {
            auto t =  std::chrono::duration_cast<std::chrono::seconds>(telapsed).count();
            std::cout << BColors::UNDERLINE << name << "\t:\t" << static_cast<int>(t/60) << "m " << t%60 << "s" << BColors::ENDC << std::endl;
        }

    private:
        Clock::time_point epoch;
        Clock::duration telapsed;
};
