from netCDF4 import Dataset as ncfile
import numpy as np
import csv
import os
import copy
from collections import OrderedDict
import re
import datetime
from tzwhere import tzwhere
import inspect
import logging
from shutil import make_archive
import fnmatch


class dataset_metadata(object):
    def __init__(self, name=None, dirc=None, start_date=None, end_date=None):
        self.name = name
        self.dirc = dirc
        self.start_date = start_date
        self.end_date = end_date
        self.var_list = None


class data_reader(object):
    ''' Object that calls external function to read vector from file '''
    def __init__(self, src_file, f):
        self.logger = logging.getLogger(__name__)
        self.readfunc = f
        self.src = src_file
        self.__labels = []

        try:
            argspec = inspect.getargspec( self.readfunc )
            arg_names = list(argspec.args)
            arg_values = list(argspec.defaults)
        except:
            self.logger.info('cant read arguments')
            arg_names = []
            arg_values = []

        while len(arg_names) > len(arg_values)
            arg_values.insert(0,None)

        self.__args = dict(zip(arg_names,arg_values))

    def set_args( self, arg_names, arg_values ):
        self.__args = dict(zip(arg_names,arg_values))

    def get_args( self ):
        return [k for k in self.__args]

    def get_argval(self, name):
        try:
            val = self.__args[name]
        except KeyError as e:
           raise Exception('{} is not in self.__args '.format(name)) 

        return val

    def update_args(self, args):
        for k,v in args.iteritems():
            if k in self.__args:
                self.__args[k] = v
            elif not self.__args:
                self.logger.info('addings function arguments not found with inspect')
                self.__args[k] = v
            else:
                self.logger.info('not adding function argument')


    def __str__(self):
        return 'data reader for file {}'.format(self.src)

class nc_vec_readaer(data_reader):
    ''' Read a netcdf file returning 1D vec from single pt or site  '''

    def __init__(self, src_file ):

        super(data_reader, self).__init__(src_file, f=ncfile)

        f = ncfile(self.src,'r'):
        self.__labels  = f.variables()
        self.__timevar = f.variables.keys()[0]
        f.close()


    def __str__(self):
        return 'csv reader for file {}'.format(self.src)

    def starttime(self , time_header = None):
        return self.timestep_timestamp( time_header = time_header )

    def endtime(self, time_header )
        return self.timestep_timestamp( timestep = -1 , time_header = time_header )

    def get_timestamp(self, timestep = 0, time_header = None):

        try:
            f = ncfile(self.src,'r')
            timevar = time_header.keys()[0]
            time_vec = f.variables(timevar)[:]
            outtime = time_vec[timestep]
            f.close()
        except:
            self.logger.info('cannot return time at timestep {}'.format(timestep))
            raise Exception('cannot return time at timestep {}'.format(timestep))

        return outtime

    def variables(self):
        try:
            f = ncfile(self.src,'r')
            variables = f.variables.keys()
            time_vec = f.variables[timevar][:]
            f.close()
        except:
            msg = 'error reading header labels from {}'.format(self.src)
            self.logger.info(msg) 
            raise Exception(msg) 
        return labels

    def gen_vec(self, vname, args = {} ):
        self.logger.info(' generating {} '.format(vname) )
        if vname in self.__labels:
            column = vname
        else:
            column = self.__labels[0]

        self.logger.info('calling nc reader with: {} '.format(str(arg)))
        try:
            f = ncfile(self.src,'r')
            vec = f.variables[column].get_values()
            f.close()
        except:
            msg = 'error reading {} from {}'.format(columnm,self.src)
            self.logger.info(msg) 
            raise Exception(msg) 

        return vec


class csv_vec_reader(data_reader):
    ''' Read a csv file returning single column as vector '''
    import inspect
    def __init__(self, src_file ):
        #set f to None as use a built in 
        super(data_reader, self).__init__(src_file, f=None)

        self.__labels  = self.variables()
        self.set_args(['delim'],[','])

    def __str__(self):
        return 'csv reader for file {}'.format(self.src)

    def starttime(self , time_header = None):
        return self.timestep_timestamp( time_header = time_header )

    def endtime(self, time_header )
        return self.timestep_timestamp( timestep = -1 , time_header = time_header )

    def get_timestamp(self, timestep = 0, time_header = None):
        n = 0
        with open(src, 'r') as f:
            if timestep == -1:
                for row in reversed(list(csv.reader(f))):
                    outtime = row[self.__label.index(time_header)]
                    break
            else:
                for row in list(csv.reader(f)):
                    outtime = row[self.__label.index(time_header)]
                    if n == timestep
                        break
                    else:
                        n = n + 1
        return outtime


    def variables(self):
        try:
            with open(self.src, 'r') as f:
                labels = [fld for fld in csv.DictReader(f).fieldnames]
        except:
            msg = 'error reading header labels from {}'.format(self.src)
            self.logger.info(msg) 
            raise Exception(msg) 
        return labels

    def gen_vec(self, vname, args = {} ):
        self.logger.info(' generating {} '.format(vname) )
        if vname in self.__labels:
            column = self.__labels.index(vname)
        else:
            column = 0

        self.logger.info('calling csv.reader with: {} '.format(str(arg)))
        self.update_args(args)
        file_reader = csv.reader(open(self.src, 'r'), delimiter=self.__args['delim'])
        _ = next(file_reader)  #skip the header
        vec = np.array([float(x[arg['usecols']]) for x in file_reader])

        return vec


#marshall dimension info
class io_dimensions(object):
    ''' Container holding dimensions for Earth Science data '''
    def __init__(self, names = ['Time'] , 
                       sizes=[None] , 
                       units=['seconds'] ):
        
        self.name = tuple( names )
        self.units = tuple( units )
        self.num = tuple(sizes)

    def get( self ):
        return tuple( [d for d in self.name] )

    def __str__(self):
        out_string = ''
        for name, num, units in zip(v.name,v.num,v.units):
            out_string = '{} ({},{},{}) \n'.format(out_string, name, num, units)

class io_variable(object):
    ''' Container to  hold vector read from a file as immutable object,
        data vector to work with and metadata about the variable.
    '''
    def __init__(self, **kwargs):
        self.name = kwargs.pop('name',('Time'))
        self.rename = kwargs.pop('rename', self.name)
        self.dims = kwargs.pop('dims',io_dimensions() )
        self.fill_value = kwargs.pop('fill_value',-9999)
        self.units = kwargs.pop('units',None)
        self.dtype = kwargs.pop('dtype',None)
        self.var_reader = kwargs.pop('var_reader', None)
        self.scale = kwargs.pop('scale', 1.0)
        self.offset = kwargs.pop('offset', 0.0)
        self.data = None
        self.raw_writeable = True
        self.logger = logging.getLogger(__name__)

        #if we have the reader get data into .__raw
        if self.var_reader is not None:
            self.logger.info('using {} to gen {} '.format(
                str(self.var_reader),self.name))
            self.logger.info('scale- {}   offset -  {} '.format(
                str(self.scale),str(self.offset)))

            in_vec = self.var_reader.gen_vec( self.name  ) 
            if 'f' in self.dtype:
                in_vec *= self.scale
                in_vec += self.offset
            else:  #ensure we do not change int to float
                in_vec *= int(self.scale)
                in_vec += int(self.offset)

            self.set_raw(in_vec)
            self.data = self.__raw.view()

            self.logger.info(' max value of {} is {} '.format(
                self.name,str(np.max(in_vec))))
            self.logger.info(' min value of {} is {} '.format(
                self.name,str(np.min(in_vec))))

    def set_raw(self, in_array):
        if self.raw_writable: 
            try:
                self.__raw = in_array.copy()
                self.__raw.flags.writeable = False
                self.raw_writeable = False
            except:
                raise Exception('set_raw') 
        else:
            msg = 'Warning: attempt to set the raw data of io_variable {} '.format(
                    self.name)
            self.logger.info(msg)


    def get_raw(self):

        try:
            return self.__raw.view()
        except:
            raise Exception("Something bad happened") 


    def set_data(self, in_array = None) :

        if in_array is None:
            try:
                self.data = self.__raw.astype(self.dtype)
            except:
                raise Exception('.__raw to data copy error')
        else:
            try:
                self.data = in_array.astype(self.dtype)
            except:
                raise Exception('in_array to data copy error')

    def get_data(self):

        if self.data is None:
            self.set_data()

        try:
            return self.data.view()
        except:
            raise Exception("Something bad happened") 


def time_int_to_datetime(time_int):
    ''' convert time_int to an str calling
        time_str_to_datetime
    '''
    return time_str_to_datetime( str( time_int ) )

def time_str_to_datetime(time_str):
    ''' time_str is the date string in the form 
    yyyymmdd yyyymmddhh yyyymmddhhmm yyyymmdd hh:mm:ss
    a datetime object at this date is output
    '''
    #remove :
    time_str = re.sub("\D", "",time_str)
    try:
        hour = int(time_str[8:10])
    except:
        hour = 1

    try:
        minut = int(time_str[10:12])
    except:
        minut = 0

    try:
        date = datetime.datetime(int(time_str[0:4]),
                                 int(time_str[4:6]),
                                 int(time_str[6:8]),
                                 hour,minut )
    except:
        date = None

    return date


def convert_local_to_utc(local_date,latitude,longitude):
    ''' create tz object using utc from a local time object
    and the lat/lon input as location[lat]=latitude,location[lon]=longitude
    location has units of degrees (south_north, west_east) '''
    tzw = tz.where.tzwhere()  #object to get zone name
    tz_local_name = tzw.tzNameAt(latitude,longitude)  #get zone
    #tz_offset = tz.utcoffset(tz_name, is_dst=True)
    tz_utc = tz.tzutc()

    return local_date.astimezone(tz_utc)



def get_outfile_name(infile,file_substring_removal):
    ''' Return string infile with each of the strings in
        the list infile,file_substring_removal removed from
        it (if possible)
    '''
    outfile = os.path.splitext(os.path.basename(infile))[0]
    #will work if infile,file_substring_removal is string!
    for remove_me in file_substring_removal:
        outfile = outfile.replace(remove_me,'')  #_F and _CORR
        print('outfile is {}'.format(outfile))
    return outfile


def is_needed(f,timeperiod):
    ''' return true if f ends with .csv and contains string
        timeperiod
    '''
    if f.endswith('.csv') and timeperiod in f:
        return True
    else:
        return False


def locate_flxnet_data(base_dir='~/',timeavg='',make_utc=True):
    ''' search directory  base_dir/country/site/data for csv files with 
        timeavg in file name return an Ordered Dict using site id's as key 
        (CN-SID where CN is country SID is three letter site name), 
        the value is full file location as a string
    '''
    #countries_list = [d for d in os.listdir(base_dir) if 
    #                   os.path.isdir(os.path.join(base_dir, d)) and 
    #                   not d.startswith('.') and 'ummar' not in d]

    #all_files=OrderedDict()
    #for cntry in countries_list:
    #    current_dir=os.path.join(base_dir, cntry)
    #    sites=[d for d in os.listdir(current_dir) 
    #            if os.path.isdir('{}/{}'.format(current_dir,d))]
    #    logger.info('{}\tsite list:{}'.format( current_dir, sites ))

    #    for site in sites:
    #        site_id = '{}-{}'.format(cntry,site)
    #        my_dir=os.path.join(current_dir,site)
    #        files = [ f for f in os.listdir(my_dir) if 
    #                       os.path.isfile(os.path.join(my_dir,f)) and 
    #                       is_needed(f,timeavg)]
    #        logger.info('\t\ti{} file list:{}'.format( site, files))
    #        for fl in files:
    #            possible_yrs = re.findall(']d+', fl ))
    #            if int(possible_yrs[0]) != 2015:
    #                logger.error('file name should be of form {}'.format(
    #                    'FLX_AR-SLu_FLUXNET2015_SUBSET_MM_2009-2011_1-3.csv'))
    #            all_files[site_id] = dataset_metadata(name = fl,
    #                    dirc = my_dir,
    #                    utc_flg = make_utc,
    #                    start_date = datetime.date(int(possible_yrs[1]), 1, 1)
    #                    end_date = datetime.date(int(possible_yrs[1]), 1, 1))

    #return all_files

    file_info = OrderedDict()
    for bdir, dirs, fnames in os.walk('./'):
        for fname in fnmatch.filter(fnames, '*.csv'):
            full_file = os.path.join(bdir, fname)
            info =  bdir.split('/')
            country = info[-2]
            site = info[-1]

            possible_yrs = re.findall(']d+', fname ))
            if int(possible_yrs[0]) != 2015:
                logger.error('file name should be of form {}'.format(
                    'FLX_AR-SLu_FLUXNET2015_SUBSET_MM_2009-2011_1-3.csv'))

            all_files[site_id] = dataset_metadata(name = fname,
                    dirc = bdir,
                    utc_flg = make_utc,
                    start_date = datetime.date(int(possible_yrs[1]), 1, 1)
                    end_date = datetime.date(int(possible_yrs[1]), 1, 1))


def sat_vapor_pressure(t_k):
    e_sat = 0.061078 * np.exp( 17.269388 * ( t_k - 273.15 ) / ( t_k - 35.86 ) )
    return e_sat

def alt_sat_vapor_pressure(t_degc):
    ''' sat vapor pressure given temp in C'''
    e_sat = 0.6108 * np.exp(17.27 * t_degc / (t_degc + 237.3))
    return e_sat

def vapor_pressure(T, vpd):
    ''' actual vapor pressure given temp in C and vpd'''
    e_a = vpd + sat_vapor_pressure(T)
    return e_a

def vpd_to_spechum(vpd,t_k,p_srf):
    ''' use vpd,T, and P (p_srf) to get spec hum '''
    e_a = vapor_pressure(t_k, vpd)
    q = 0.622 * e_a / p_srf
    return q



#test_file='../US/ARc/FLX_US-ARc_FLUXNET2015_SUBSET_HH_2005-2006_1-3.csv'
#hrldas forcing names: T3D, Q2D, U2D, V2D, PSFC(pa), RAINRATE(mm/s),SWDOWN,LWDOWN
# dims: Time, south_north, west_east (1,1,1)

#output is outname = outdir+yyyy_start+nums(mm_start)+nums(dd_start)+nums(hh_start)+".LDASIN_DOMAIN1"

def convert_flxnet_csv_to_hrldas_forcing(
        infile,outdir,out_dims=['Time','south_north','west_east'],
        flx_vnames = ['TA_F','VPD_F','SW_IN_F','LW_IN_F','WS_F','PA_F','P_F'],
        hrldas_vnames = ['T2D','Q2D','SWDOWN','LWDOWN','U2D','PSFC','RAINRATE'],
        out_units = ['K','kg/kg','W/m2','W/m2','m/s','Pa','mm/s'],
        dim_names = ['Time','south_north','west_east'],
        dim_sizes=[None,1,1],
        dim_units=['seconds since','degrees_north','degrees_east'],
        convert_to_utc=False, def_otype='float32' ):

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    output_tally = []

    #reader object that read one var at a time from csv
    var_reader = csv_vec_reader( infile , np.genfromtxt )

    #make dimensions
    io_dims = io_dimensions(dim_names, dim_sizes, dim_units)

    # make dict containing csv_io_var 
    io_vars = {}
    for iname,oname,unit in zip(flx_vnames,hrldas_vnames,out_units):
        scale = 1.0
        offset = 0.0
        if 'RAIN' in oname:
            scale = 1.0 / 1800.0  #mm/s from mm over timestep
        elif 'T2D' in oname:
            offset = 273.15
        elif 'PSFC' in oname:
            scale = 1000.0
        elif 'VPD' in oname:
            scale = 1000.0

        logger.info('creating {} '.format(iname))
        io_vars[iname] = io_variable(name = iname,
            rename = oname, units = unit, src= infile, scale = scale,
            offset = offset, dims = io_dims, dtype = def_otype, 
            var_reader = var_reader )
        if 'WS' in iname:
            offset = 0.0
            scale = 0.0
            io_vars['V2D'] = io_variable(name = 'WS_F',
                rename = 'V2D', units = unit, src= infile, scale = scale,
                offset = offset, dims = io_dims, dtype = def_otype,
                var_reader = var_reader )

    time_name = 'Time'

    itime_vec = var_reader.gen_vec(0).astype('int')
    otime_vec = itime_vec[0::2].copy()   #hourly but local

    #create the output data using input
    #first need sepc humidity from vpd,temperature, and psrf
    #views to simplify code
    vpd_vec = io_vars['VPD_F'].get_data()
    t_vec = io_vars['TA_F'].get_data()
    p_vec = io_vars['PA_F'].get_data()

    q_srf = vpd_to_spechum(vpd_vec,t_vec,p_vec) 
    io_vars['VPD_F'].set_data( q_srf )

    #change key so key is output name
    io_vars['Q2D'] = io_vars.pop('VPD_F')

    #second need to get hourly from half hourly
    for _,var in io_vars.iteritems():
        vec = var.get_data()
        out_vec = np.mean(vec[:(len(vec)//2)*2].reshape(-1, 3), axis=1)
        var.set_data(out_vec)

    #list of time slices to do
    #to use mpi and N tasks for a subset of the list
    #just subset below list using myrank
    init_datetime = time_int_to_datetime(otime_vec[0])

    todo_list = [i for i in otime_vec ]

    #loop so every time step produces 1 output file
    #that contains data for a single time slice
    for itime, my_time in enumerate(todo_list):

        current_date = time_int_to_datetime( my_time )
        if convert_to_utc:
            current_date = convert_local_to_utc(current_date, lat, lon)

        #add .nc.  hrldas uses input without any file type suffix
        #change name at end
        outfile = '{}/{}.LDASIN_DOMAIN1.nc'.format(
            outdir, current_date.strftime('%Y%m%d%H') )
        if os.path.isfile(outfile):
            os.remove(outfile)

        if not os.path.isdir(outdir):
            os.makedirs(outdir)

        ncout=ncfil(outfile,'w')
   
        nc_dims = [ncout.createDimension(dname, dsize ) 
                     for dname,dsize in zip(io_dims.name, io_dims.num) ]

        #can make time variable but not needed
        #nc_time = ncout.createVariable(vname,def_otype,dim_tup,fill_value = -9998)
        made_vars = []
        for k,v in io_vars.iteritems():

            out_vec = v.get_data()

            made_vars.append( ncout.createVariable(v.rename,
                v.dtype, v.dims.get(), fill_value = v.fill_value ) )  #v.fill_value
            made_vars[-1].units = v.units
            made_vars[-1][0,0,0] = out_vec[itime]

        #close file
        ncout.close()

        #finally remove .nc from filename
        discard_extension = outfile.replace('.nc','')
        os.rename(outfile,discard_extension)

        output_tally.append(discard_extension)

    return output_tally

#below is terribles async example
#asyncio best suited to writing 1000s of output hrldas nc files
#import asyncio
#
#async def site_processor(site_file,outdir,outformat, utc_flg, st, ed):
#    """thread worker function"""
#    if outformat.lower() == 'hrldas':
#        made_file = convert_flxnet_csv_to_hrldas_forcing( 
#                site_file, outdir, utc_flg, st, ed )
#    elif outformat.lower() == 'nc':
#        pass
#    #return made_file
#
#async def task_creator(site_file,outdir,outformat, utc_flg, st, ed):
#    task = asyncio.create_task(
#            site_processor(site_file,outdir,outformat, utc_flg, st, ed))
#
#    # "task" can now be used to cancel "nested()", or
#    # can simply be awaited to wait until it is complete:
#    # progress updates?
#    await task
#
#if __name__ == '__main__':
#    start = time.time()  
#    loop = asyncio.get_event_loop()
#    
#    tasks = [asyncio.ensure_future(
#        task_creator(site_file,outdir,outformat, utc_flg, st, ed)
#        ) for site_id, site_file,[
#            outdir,outformat, utc_flg, st, ed] in data_files.iteritems()]
#    ]
#    loop.run_until_complete(asyncio.wait(tasks))  
#    loop.close()
#    
#    end = time.time() 
#
#    for site_id,[site_file,utc_flg,st,ed] in data_files.iteritems():
#        asyncio.run(task_creator(site_file,outdir,outformat, utc_flg, st, ed))
#

# process the files for each site using multiprocessing
# limit threads to ncpus - 1 using pool of work
# enables transition to processing gridded data
# if each grid cell is treated as a site
def WorkerFunc(site_file,outdir,outformat, utc_flg, st, ed):
    """thread worker function"""
    if outformat.lower() == 'hrldas':
        convert_flxnet_csv_to_hrldas_forcing( site_file, outdir, utc_flg, st, ed )
    return

if __name__ == '__main__':
    jobs = []
    for site_id,[site_file,utc_flg,st,ed] in data_files.iteritems():
        p = multiprocessing.Process(
                target=WorkerFunc, args=(site_file,outdir,utc_flg,st,ed))
        jobs.append(p)
        p.start()

    #now I have a list of jobs
    #do I yield jobs?

from __future__ import division
import sys

for i, _ in enumerate(p.imap_unordered(do_work, xrange(num_tasks)), 1):
    sys.stderr.write('\rdone {0:%}'.format(i/num_tasks))

import multiprocessing
from functools import partial

data_list = [1, 2, 3, 4]

def prod_xy(x,y):
    return x * y

def pool_and_map(func_for_mp, func_args):
    '''map func_for_mp with values of data_files
    across total number of cpu's '''
    data_list = [v for _,v in func_args.iteritems()]
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    return pool.map(func_for_mp, data_list)

if __name__ == '__main__':
    parallel_runs(data_list)
    #monitor or return jobs?


if __name__ == "__main__":

    logger = logging.getLogger(__name__)
    handler = logging.FileHandler('debug_log')
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    base_outdir = '/home/thedude/research/Fluxnet/flxnet_hrldas/'
    input_dir='/home/thedude/research/Fluxnet/2015/Tier2/flxnet'
    file_strs = []#['_2-3','_1-3','_SUBSET','_HH']
    var_strs = ['_CORR']
    
    data_files = locate_flxnet_data(input_dir,'HH')
    sites = data_files.keys()  

    func_args = {}
    force_utc = True
    for site_id,site_file in data_files.iteritems():
        out_dir = os.path.join(base_outdir,site_id)
        func_args[site_id] = [site_file, out_dir, force_utc,
                usr_dates[site_id][s],usr_dates[site_id][e]]


    to_do = [k for k in data_files.keys() if 'US' in k.upper()]
  
    for site_id,site_file in data_files.iteritems():

        logger.info('Processing {} using {}'.format(site_id, site_file))
        outdir = os.path.join(base_outdir, str(site_id) )
        made_files = convert_flxnet_csv_to_hrldas_forcing( site_file, outdir )
        archive_name = '{}'.format(site_id)
        make_archive( archive_name, 'gztar', outdir  )
        os.rename(
                '{}.tar.gz'.format(archive_name), '{}/{}.tar.gz'.format(
                    base_outdir, archive_name) )

