##PBS -l  nodes=1:gen9:ppn=2

cd $PBS_O_WORKDIR
./my_application
