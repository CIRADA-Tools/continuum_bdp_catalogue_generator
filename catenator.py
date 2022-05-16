############################################################
#
#    * * *   I M P O R T S    * * *
#

# notes: vi spell-checker
# [1] http://thejakeharding.com/tutorial/2012/06/13/using-spell-check-in-vim.html

# system tools
import os
import sys
sys.path.insert(0,"media")

# process management
import atexit
import psutil
import enlighten
import traceback

# configuration
import shelve
import yaml as yml
from types import SimpleNamespace

# RE: PyYAML yaml.load(input) Deprecation (https://msg.pyyaml.org/load)
yml.warnings({'YAMLLoadWarning': False})

# file handling
import glob
import json
# vospace notes:
# [1] https://www.canfar.net/storage/list/cirada/continuum/mboyce/pipeline_upload_test
# [2] cf., repo file: Continuum_common/downloader/catalogues/downloader.py
from vos import Client as VOSpace
from shutil import copyfile

# astro tools
import numpy as np
import pandas as pd
from astropy.io import ascii
from astropy import units as u
from astropy.table import Table
from astropy.table import QTable
from astropy.table import vstack

# pipeline tools
import re
import time
import urllib
import bdsf as bd
from modules import *
from libs import Timer
from libs import TimeStats
from libs import get_time_string
from libs import DuplicatesChecker

# command line interface
# TO-DO: Investigate shell completion (https://click.palletsprojects.com/en/7.x/bashcomplete/)
import click



############################################################
#
#    * * *   P I P E L I N E   S T A T E S   * * *
#

# configuration: persistent data store
# notes: https://docs.python.org/3/library/shelve.html
state_cache = '.cache/configuration'

# processing states
# notes: 
# [1] https://treyhunner.com/2019/04/why-you-shouldnt-inherit-from-list-and-dict-in-python/
# [2] https://stackoverflow.com/questions/3738381/what-do-i-do-when-i-need-a-self-referential-dictionary
# [3] https://stackoverflow.com/questions/50841165/how-to-correctly-wrap-a-dict-in-python-3
# [4] https://stackoverflow.com/questions/1305532/convert-nested-python-dict-to-object
pipeline_states = {
    'download'    : "VLASS Subtile downloading stage.",
    'process'     : "PyBDSF image processing stage.",
    'vlad'        : "Subtile-based VLAD catalogue creation stage.",
    'diagnostics' : {
        'diagnostics'        : "Creation of information for QA flag determination.",
        'diagnostic_plots'   : "Diagnostic plots and test HTML creation stage.",
        'vlad_compilate'     : "VLAD subtile-based catalogue stacking and update stage.",
        'subtile_info_table' : "Subtile Info. Table creation stage.",
    },
    'qa'          : "VLAD QA flags update and BDP deployment.",
    'upload'      : "Upload BDPs to CIRADA DB.",
}

# process lock: persistent data store
process_lock = '.lock/process_lock'

# process locking and monitoring helper-class
class ProcessLock:
    def __init__(self):
        self.lock_db = self.__init_process_lock(process_lock)
        self.pid_key = 'PID'
        self.handle = None

    def __init_process_lock(self,process_lock_file):
        lock_dir = re.sub(r"^((.*?/)*).*$",r"\1",process_lock_file)
        if lock_dir != '' and not os.path.isdir(lock_dir):
            print(f"Creating process-lock dir: {lock_dir}")
            os.makedirs(lock_dir)
            print(f"Creating process-lock db: {process_lock_file}")
            with shelve.open(process_lock_file):
                pass
        return process_lock_file

    def lock(self,pid):
        if isinstance(pid,int):
            with shelve.open(self.lock_db) as db:
                db[self.pid_key]=pid
        return self

    def unlock(self):
        with shelve.open(self.lock_db) as db:
            if self.pid_key in db:
                del db[self.pid_key]
        return self

    def is_locked(self):
        if self.get_pid() is None:
            return False
        return True

    def get_pid(self):
        with shelve.open(self.lock_db) as db:
            if self.pid_key in db:
                pid = db[self.pid_key]
                if psutil.pid_exists(pid):
                    return pid
                else:
                    del db[self.pid_key]
        return None

    def __enter__(self):
        self.handle = shelve.open(self.lock_db,writeback=True)
        return self.handle

    def __exit__(self,exc_type,exc_value,tracback):
        self.handle.close()
        self.handle = None



############################################################
#
#    * * *   P I P E L I N E   H I E R A R Y   * * *
#
#
# This is the core of the code which handles the pipeline stages
# through a class hierarchical structure.
#
#            Class             Function
#            -----             --------
#                 Config: Configuration
#         Logger(Config): Logging
#     Downloader(Logger): Downloading
#    Process(Downloader): PyBDSF image processing
#          VLAD(Process): VLAD Catalogue handling
#      Diagnostics(VLAD): QA diagnostic tools
#        QA(Diagnostics): QA parameters / flagging
#             Upload(QA): VOSpace uploading
#

class Config:
    def __init__(self):
        self.config = self.__load_config()
        self.directories           = self.config['directories']
        self.downloads             = self.config['downloads']
        self.pybdsf                = self.config['pybdsf']
        self.vlad                  = self.config['vlad']
        self.diagnostics           = self.config['diagnostics']
        self.diagnostics_html_dir  = f"{self.directories['diagnostics']}/html"
        self.diagnostic_plots_dir = f"{self.directories['diagnostics']}/plots"
        self.qa                    = self.config['qa']
        self.upload                = self.config['upload']
        self.__make_directories()
        self.time_stamp = time.strftime('%Y%m%d_%H%M%S')
        self.__init_runtime_states()

    def __init_runtime_states(self):
        with ProcessLock() as db:
            for state in pipeline_states.keys():
                if state in db.keys():
                    del db[state]

    def get_timestamp(self,is_now=False):
        return time.strftime('%Y%m%d_%H%M%S') if is_now else self.time_stamp

    def __load_config(self):
        configuration_file = self.get_configuration_file()
        print(f"Using pipeline config: {configuration_file}")
        config = yml.load(open(configuration_file))
        return config

    def __make_directories(self):
        print("Setting up infrastructure directories...")
        print(self.directories.keys())
        for catagory in self.directories.keys():
            directory = self.directories[catagory]
            if not os.path.isdir(directory):
                self.print(f"> Creating {catagory} dir: {directory}")
                os.makedirs(directory)
        if not os.path.isdir(self.diagnostics_html_dir):
             self.print(f"> Creating diagnostics/html dir: {self.diagnostics_html_dir}")
             os.makedirs(self.diagnostics_html_dir)
        if not os.path.isdir(self.diagnostic_plots_dir):
             self.print(f"> Creating diagnostics/html dir: {self.diagnostic_plots_dir}")
             os.makedirs(self.diagnostic_plots_dir)
        print("[Done]")

    def get_configuration_file(self):
        with shelve.open(state_cache) as cache:
            configuration_file = f"{cache['config_dir']}/{cache['config_file']}"
        return configuration_file

    def get_directories(self):
        return self.directories

    def get_diagnostics_html_dir(self):
        return self.diagnostics_html_dir

    def get_diagnostic_plots_dir(self):
        return self.diagnostic_plots_dir

    def get_manifest(self):
        return self.downloads['manifest']

    def get_pybdsf_settings(self):
        return self.pybdsf

    def get_vlad_settings(self):
        return self.vlad

    def get_diagnostic_settings(self):
        return self.diagnostics

    def get_qa_settings(self):
        return self.qa

    def get_upload_settings(self):
        return self.upload


class Logger(Config):
    def __init__(self,is_log=True):
        # TO-DO: We're gonna have to pipe to stdout and stderr
        # std.out = open(log_file,'w')
        # before call to init -- caution: make sure log_dir exists...
        Config.__init__(self)
        self.log_file = self.__get_log_filename()
        if is_log:
            print(f"Redirecting output to: {self.log_file}")
            with ProcessLock() as db:
                db['log_file'] = self.log_file
            self.fh = open(self.log_file,'w')
            sys.stdout = self.fh
            sys.stderr = self.fh
            self.print(f"My Process ID: {os.getpid()}")
            self.print(f"Using pipeline config: {self.get_configuration_file()}")
        else:
            with ProcessLock() as db:
                if 'log_file' in db.keys():
                    del db['log_file']
            self.fh = None

    def __get_log_filename(self):
        prefix = f"{self.directories['logs']}/{self.__class__.__name__.lower()}"
        log_files = glob.glob(f"{prefix}_*.log")
        version_no = 1
        if len(log_files)>0:
            try:
                version_no = max([int(re.sub(r"^.*?(\d+).*$",r"\1",f)) for f in log_files])+1
            except:
                pass
        return f"{prefix}_{version_no}.log"

    def flush_logs(self,is_prune=False):
        self.print(f"{'Pruning' if is_prune else 'Flushing'} log files...")
        log_files = glob.glob(f"{self.directories['logs']}/{self.__class__.__name__.lower()}_*.log")
        if len(log_files) > 0:
            keep = None
            if is_prune:
                n_last = max([int(re.sub(r"^.*?(\d+).*$",r"\1",f)) for f in log_files])
                for log_file in log_files:
                    if re.search(f"{n_last}\.",log_file):
                        keep = log_file
                        break
            for log_file in log_files:
                if log_file != keep and os.path.isfile(log_file):
                    self.print(f"> Flushing: {log_file}")
                    os.remove(log_file)
            if not keep is None:
                self.print(f"> Kept: {keep}")
        self.print("[DONE]")
        return self

    def print(self,text):
        #if self.fh is None:
        #    print(text)
        #else:
        #    self.fh.write(f"{text}\n")
        print(text)
        return self


class Downloader(Logger):
    def __init__(self,is_log=True):
        Logger.__init__(self,is_log)
        self.urls = Table.read(self.downloads['manifest'],format='csv')['file']
        self.dups = DuplicatesChecker(self.downloads['manifest'])
        self.tile_dirs = self.__make_download_dirs(self.urls)
        self.max_attempts = 5
        self.sleep = 10

    def __get_tile_name(self,url):
        return re.sub(r"^.*?(T\d\dt\d\d).*$",r"\1",url)

    def __get_tile_dir(self,url):
        return f"{self.directories['tiles']}/{self.__get_tile_name(url)}"

    def __make_download_dirs(self,urls):
        self.print("Setting up VLASS download directories...")
        tile_dirs = set(self.__get_tile_dir(url) for url in urls)
        for tile_dir in tile_dirs:
            if not os.path.isdir(tile_dir):
                self.print(f"> Creating tile dir: {tile_dir}")
                os.makedirs(tile_dir,exist_ok=True)
        self.print("[Done]")
        return tile_dirs

    def __get_subtile(self,url):
        out_file = f"{self.__get_tile_dir(url)}/{re.sub(r'^(.*?/)+','',url)}"
        self.print(f"Processing: {url}")
        is_error     = False
        is_duplicate = False
        if os.path.isfile(out_file):
            self.print(f"> Already have '{out_file}', skipping...")
        elif self.dups.is_keep(url):
            self.print(f"> Creating: {out_file}")
            if re.search("^\s*file://",url):
                in_file = re.sub(r"^\s*file://","",url)
                self.print(f"> $ cp {in_file} {out_file}")
                try:
                    copyfile(in_file,out_file)
                except Exception as e:
                    self.print(f"> ERROR: {e}")
                    is_error = True
            else:
                attempts = 0
                while attempts < self.max_attempts:
                    try:
                        response = urllib.request.urlopen(url)
                        content  = bytearray(response.read())
                        with open(out_file,'wb') as f:
                            f.write(content)
                        break
                    except Exception as e:
                        self.print(f"> ERROR: {e}: Trying again in {self.sleep}s...")
                        attempts += 1
                    time.sleep(self.sleep)
                if attempts >= self.max_attempts:
                    self.print(f"> ERROR: Exceeded max tries of {self.max_attempts}.")
                    is_error = True
        else:
            self.print(f"> Older Version: '{url}', skipping...")
            is_duplicate = True
        return {'is_error': is_error, 'is_duplicate': is_duplicate}

    def get_duplicates_checker(self):
        return self.dups

    def fetch(self,is_flush=False):
        with ProcessLock() as db:
            db['download'] = {
                'PID': os.getpid(),
                'Status': "[Running]" 
            }
        if is_flush:
            self.flush(is_all=False)
        self.print(f"Fetching all {len(self.urls)} VLASS Subtiles...")
        tracker = TimeStats(len(self.urls)).start()
        error_cnt = 0
        duplicates_cnt = 0
        def update():
            s = tracker.update_dt().get_stats()
            s['Errors']     = error_cnt
            s['Duplicates'] = duplicates_cnt
            s['Status']     = "[Running]"
            with ProcessLock() as db:
                db['download'] = s
            self.print("> - [STATS] DOWNLOAD:\n> - "+"\n> - ".join([f"{re.sub(r'_',' ',k)}: {s[k]}" for k in s]))
        for url in self.urls:
            status = self.__get_subtile(url)
            if status['is_error']:
                error_cnt += 1
            if status['is_duplicate']:
                duplicates_cnt += 1
            update()
        with ProcessLock() as db:
            db['download']['Status'] = "[Done]"
        self.print("[Done]")
        return self

    def get_tile_dirs(self):
        return self.tile_dirs

    def get_subtiles(self):
        fits_files = list()
        for tile_dir in self.tile_dirs:
            fits_files.extend(glob.glob(f"{tile_dir}/*.fits"))
        return fits_files

    def flush(self,is_all=False):
        if is_all:
            self.print(f"Flushing tile directories...")
            for tile_dir in self.tile_dirs:
                files = glob.glob(f"{tile_dir}/*")
                if len(files) > 0:
                    for file in files:
                        if os.path.isfile(file):
                            self.print(f"> Flushing: {file}")
                            os.remove(file)
        else:
            self.print(f"Flushing FITS download files...")
            for tile_dir in self.tile_dirs:
                files = glob.glob(f"{tile_dir}/*.fits")
                if len(files) > 0:
                    for file in files:
                        if os.path.isfile(file):
                            self.print(f"> Flushing: {file}")
                            os.remove(file)
        self.print("[Done]")
        return self

class Process(Downloader):
    def __init__(self,is_log=True):
        Downloader.__init__(self,is_log)
        pybdsf = self.get_pybdsf_settings()
        self.processing = {
            'rms_box':  (pybdsf['processing']['rms_box']['box_size_pixels'],pybdsf['processing']['rms_box']['step_size_pixels']),
            'frequency': pybdsf['processing']['frequency'],
        }
        self.csv_file = {
            'format':       pybdsf['catalogue']['format'],
            'catalog_type': pybdsf['catalogue']['catalog_type'],
            'incl_empty':   pybdsf['catalogue']['incl_empty'],
        }
        self.region_file = {
            'format':       pybdsf['region_file']['format'],
            'catalog_type': pybdsf['region_file']['catalog_type'],
            'incl_empty':   pybdsf['region_file']['incl_empty'],
        }

    def __cleanup_csv_file_header(self,csv_file):
        if os.path.isfile(csv_file):
            with open(csv_file,"r+") as f:
                file_contents = list()
                is_keep = False
                for line in f:
                    if re.match(r"^\# Source_id",line):
                        line =  re.sub(r"^# ","",line)
                        is_keep = True
                    line = re.sub(r" +","",line)
                    if is_keep:
                        file_contents.append(line)
                f.seek(0)
                f.truncate(0)
                for line in file_contents:
                    f.write(line)
    
    def __cleanup_reg_file(self,reg_file):
        if os.path.isfile(reg_file):
            with open(reg_file,"r+") as f:
                file_contents = list()
                for line in f:
                    if re.search(r"color=",line):
                        color = 'red'
                        line = re.sub(r"(color=)\w+?(\s)",r"\1%s\2" % color,line)
                    elif re.match(r"ellipse",line):
                        # remove the region labels
                        line = re.sub(r"\s*#.*","",line)
                    elif re.match(r"point",line):
                        # remove the region labels
                        line = re.sub(r"text=.*$","color=blue",line)
                    file_contents.append(line)
                f.seek(0)
                f.truncate(0)
                for line in file_contents:
                    f.write(line)

    def process_subtile(self,fits_file):
        # bail if csv minicatalogue already exists
        csv_file = re.sub(r"\.fits$",".csv",fits_file)
        reg_file = re.sub(r"\.csv$",".reg",csv_file)
        if os.path.isfile(csv_file) and os.path.isfile(reg_file):
            csv_reg_file = re.sub(r'\.csv$','.{csv|reg}',csv_file)
            self.print(f"> Already have {csv_reg_file}, skipping...")
            return
    
        # start the timer
        clock = Timer().start()
    
        try:
            # process fits_file using rms_box method
            self.print(f"> Processing: {fits_file}")
            img = bd.process_image(
                input      = fits_file,
                rms_box    = self.processing['rms_box'],
                frequency  = self.processing['frequency']
            )
    
            # create csv catalog
            self.print(f"> Creating CSV File: {csv_file}")
            img.write_catalog(
                outfile      = csv_file,
                format       = self.csv_file['format'],
                catalog_type = self.csv_file['catalog_type'],
                incl_empty   = self.csv_file['incl_empty'],
                clobber      = True
            )
            self.__cleanup_csv_file_header(csv_file)
    
            # create reg file
            self.print(f"> Creating DS9 Region File: {reg_file}")
            img.write_catalog(
                outfile      = reg_file,
                format       = self.region_file['format'],
                catalog_type = self.region_file['catalog_type'],
                incl_empty   = self.region_file['incl_empty'],
                clobber      = True
            )
            self.__cleanup_reg_file(reg_file)
        except Exception as e:
            err_str  = "> ERROR: A problem occurred during PyBDSF image processing:"
            err_str += "\n> TRACEBACK:\n>> %s" % "\n>> ".join(traceback.format_exc().splitlines())+"\n"
            err_str += "> Skipping..."
            self.print(err_str)
    
        # stop the timer and print the results
        self.print(f"> Time: {clock.get_time_string()}")

    def process(self,is_flush=False,is_pull=False):
        with ProcessLock() as db:
            db['process'] = {
                'PID': os.getpid(),
                'Status': "[Standby]" if is_pull else "[Running]" 
            }
        if is_flush:
            Process.flush(self)
        if is_pull:
            if is_flush:
                Downloader.flush(self)
            Downloader.fetch(self)
        self.print("Creating VLASS Minicatalogues...")
        with ProcessLock() as db:
            db['process']['Status'] = "[Running]" 
        subtiles = self.get_subtiles()
        tracker = TimeStats(len(subtiles)).start()
        def update():
            s = tracker.update_dt().get_stats()
            s['Status'] = "[Running]"
            with ProcessLock() as db:
                db['process'] = s
            self.print("> - [STATS] PROCESSING:\n> - "+"\n> - ".join([f"{re.sub(r'_',' ',k)}: {s[k]}" for k in s]))
        for fits_file in subtiles:
            self.process_subtile(fits_file)
            update()
        with ProcessLock() as db:
            db['process']['Status'] = "[Done]"
        self.print("[Done]")
        return self

    def flush(self):
        self.print("Flushing PyBDSF processing files...")
        for suffix in ['subim.fits.pybdsf.log','subim.csv','subim.reg']:
            files = glob.glob(f"{self.get_directories()['tiles']}/*/*.{suffix}")
            if len(files) > 0:
                for file in files:
                    if os.path.isfile(file):
                        self.print(f"> Flushing: {file}")
                        os.remove(file)
        self.print("[Done]")
        return self

class VLAD(Process):
    def __init__(self,is_log=True):
        Process.__init__(self,is_log)
        vlad = self.get_vlad_settings()
        self.vlad_metrics = vlad['metrics']
        self.catalogue    = vlad['catalogue']
        self.field_meta   = self.__get_field_meta()
        self.modules      = self.__load_modules()

    def __get_field_meta(self):
        field_block = ['field_old','units_old','units']
        meta = self.get_vlad_settings()['catalogue']['meta']
        field_meta = dict()
        for record in meta:
            name = list(record.keys())[0]
            header = dict()
            for field in record[name].keys():
                if field in field_block:
                    header[field] = record[name][field]
            field_meta[name] = header
        return field_meta

    def __load_modules(self):
        modules = dict()
        for record in self.get_vlad_settings()['catalogue']['modules']:
            name = list(record.keys())[0]
            modules[name]=eval(record[name]['module'])
        return modules

    def get_html_help_dict(self,table_meta):
        html_keywords = ['mouseover','expanded']
        html_abreviations = {'deg': '&deg;', 'arcmin': '&prime;', 'arcsec': '&Prime;'}
        html_help = dict()
        #for field in self.catalogue['meta']:
        for field in table_meta:
            name = list(field.keys())[0]
            meta = field[name] 
            context = dict()
            if 'html_help' in meta.keys():
                local_meta = meta.copy()
                for key in meta['html_help'].keys():
                    if key == 'HTML_INCLUDE_VLAD_NAMESPACE': 
                        vlad = self.get_vlad_settings()
                        namespace = eval("vlad['"+"']['".join(meta['html_help']['HTML_INCLUDE_VLAD_NAMESPACE'].split('::'))+"']")
                        for key in namespace.keys():
                            local_meta[key] = namespace[key]
                    elif key == 'HTML_INCLUDE_INLINE_MEDIA':
                        for media in meta['html_help'][key].keys():
                            file = meta['html_help'][key][media]
                            with open(file,"r") as f:
                                local_meta[media] = f.read()
                    elif not key in html_keywords:
                        local_meta[key] = meta['html_help'][key]
                def do_it(m):
                    p = SimpleNamespace(**local_meta)
                    parameters = m.group(1).split('|')
                    parameter = eval(f"p.{parameters[0]}")
                    if len(parameters) > 1:
                        directive = parameters[1]
                        if directive == 'abbreviate' and parameter in html_abreviations.keys():
                            parameter = html_abreviations[parameter]
                    return f"{parameter}"
                if 'mouseover' in meta['html_help']:
                    context['mouseover'] = re.sub(r"{\s*([\w\|]+)\s*}",do_it,meta['html_help']['mouseover'])
                if 'expanded' in meta['html_help']:
                    context['expanded'] = re.sub(r"\{\s*(\w+)\s*\}",do_it,meta['html_help']['expanded'])
            html_help[name] = context
        return html_help

    def dump_json_vlad_html_help(self):
        json_file = f"{self.get_directories()['products']}/{self.catalogue['json_help_file']}"
        json_file = re.sub(r"\.json$",f"_{self.get_timestamp()}.json",json_file)
        self.print(f"Dumping VLAD JSON BDP help meta file: {json_file}")
        with open(json_file,"w+") as f:
            contents = json.dumps(self.get_html_help_dict(self.catalogue['meta']),indent='   ')
            f.write(contents)
        self.print("[Done]")
        return self

    def get_html_help_table(self,table_meta):
        html_help = self.get_html_help_dict(table_meta)
        html_table = list()
        style_css = "border: 1px solid black"
        th_settings = f"style=\"{style_css}\" align=\"left\" valign=\"top\""
        td_settings = f"style=\"{style_css}\""
        html_table.append(f"<table style=\"{style_css}\" cellpadding=\"5px\">")
        for field in html_help:
             description = ""
             if 'mouseover' in html_help[field].keys():
                 description += html_help[field]['mouseover']
             if 'expanded' in html_help[field].keys():
                 description += html_help[field]['expanded']
             html_table.append(f"   <tr>")
             html_table.append(f"      <th {th_settings}>{field}</th><td {td_settings}>{description}</td>")
             html_table.append(f"   </tr>")
        html_table.append(f"</table>")
        html_table = "\n".join(html_table)
        return html_table
    
    def get_vlad_html_help_table(self,is_add_div_wrapper=True):
        vlad_table = self.get_html_help_table(self.catalogue['meta'])
        if is_add_div_wrapper:
            vlad_table = [
                f"\n<div style=\"padding-left:10px\">\n",
                re.sub(r"\n","\n"+(3 * " "),"\n"+vlad_table),
                f"\n</div>\n"
            ]
            vlad_table = re.sub(r"\n+","\n","\n".join(vlad_table)+"\n")
        return vlad_table

    def convert_to_astropy_units(self,units):
        return eval(re.sub(r"(\w+)",r"u.\1",units))
    
    def __get_rename_dict(self):
        renames = dict()
        meta = self.field_meta
        for name in meta.keys():
            if 'field_old' in meta[name].keys():
                renames[meta[name]['field_old']] = name
        return renames

    def __load(self,pybdsf_out_csv_file):
        # load meta data and pybdsf subtile-image-processing output table
        meta = self.field_meta
        t_pybdsf = QTable.read(pybdsf_out_csv_file,format='csv')

        # rename fields
        renames = self.__get_rename_dict()
        for name in renames.keys():
            if name in t_pybdsf.colnames:
                t_pybdsf.rename_column(name,renames[name])
            else:
               self.print(f"WARNING: column name '{name}' not in '{pybdsf_out_csv_file}'... could be a typo in the config file.")

        # get ordered list of extract from processed table (t_pybdsf)
        keys_filter = list()
        keys_intersection = list(set(meta.keys()) & set(t_pybdsf.colnames))
        for name in meta.keys():
            for key in keys_intersection:
                if key == name:
                    keys_filter.append(key)
                    break

        # build new table with converted units
        t_filtered = QTable()
        for field in keys_filter:
            t_filtered[field] = t_pybdsf[field]
            units = None
            if 'units' in meta[field].keys():
                if 'units_old' in meta[field].keys():
                    try:
                        units = self.convert_to_astropy_units(meta[field]['units_old'])
                    except:
                        self.print(f"WARNING: '{meta[field]['units_old']}' invalid astropy unit.")
                        units = None
                    try:
                        t_filtered[field] = (t_filtered[field] * units).to(self.convert_to_astropy_units(meta[field]['units']))
                    except:
                        self.print(f"WARNING: '{meta[field]['units']}' invalid astropy unit.")
                        units = None
                else:
                    try:
                        units = self.convert_to_astropy_units(meta[field]['units'])
                    except:
                        self.print(f"WARNING: '{meta[field]['units']}' invalid astropy unit.")
                        units = None
                    t_filtered[field] = t_filtered[field] * units

        return t_filtered

    def __process_modules(self,pybdsf_table,meta,metrics,csv_file):
        vlad_table = QTable()
        for record in self.catalogue['meta']:
            name = list(record.keys())[0]
            if name in self.modules.keys():
                vlad_table[name] = self.modules[name](pybdsf_table,meta,metrics,csv_file)
            elif name in pybdsf_table.keys():
                vlad_table[name] = pybdsf_table[name]
        return vlad_table

    def __write_peak_to_ring_reg_file(self,vlad_table,csv_file):
        reg_file = re.sub(r"\.csv$",".peak_to_ring.reg",csv_file)
        r_core = (5 * u.arcsec).value
        r_ring = (10 * u.arcsec).value

        self.print(f"> Creating Peak_to_ring region file: {reg_file}")
        contents = [\
          '# Region file format: DS9 version 4.1\n' + \
          'global color="green" font="helvetica 10 normal" select=1 highlite=1 edit=1 move=1 delete=1 include=1 fixed=0 source\n' + \
          'fk5']
        for r,d,m in zip(vlad_table['RA'].value,vlad_table['DEC'].value,np.round(vlad_table['Peak_to_ring'],3)):
            contents.append(f'ellipse({r},{d},{r_core}",{r_core}",0)'+"\n"+f'ellipse({r},{d},{r_ring}",{r_ring}",0)#text="{m}" color="yellow"')
        contents="\n".join(contents)
        with open(reg_file,'w+') as f:
            f.write(contents)

    def build(self,is_flush=False,is_pull=False):
        with ProcessLock() as db:
            db['vlad'] = {
                'PID': os.getpid(),
                'Status': "[Standby]" if is_pull else "[Running]" 
            }
        if is_flush:
            VLAD.flush(self)
        if is_pull:
            Process.process(self,is_flush,is_pull)

        # convert meta list to dict
        meta = dict()
        for record in self.catalogue['meta']:
            name = list(record.keys())[0]
            meta[name] = record[name]

        # get metrics
        metrics = self.vlad_metrics

        # process pybdsf output csv files
        self.print("Processing PyBDSF output CSV files...")
        processed_files = glob.glob(f"{self.get_directories()['tiles']}/*/*subim.csv")
        tracker = TimeStats(len(processed_files)).start()
        def update():
            s = tracker.update_dt().get_stats()
            s['Status'] = "[Running]"
            with ProcessLock() as db:
                db['vlad'] = s
            self.print("> - [STATS] VLAD:\n> - "+"\n> - ".join([f"{re.sub(r'_',' ',k)}: {s[k]}" for k in s]))
        vlad_table = None
        for processed_file in processed_files:
            out_file = re.sub(r"(\.csv)$",r".vlad\1",processed_file)
            if os.path.isfile(out_file):
                self.print(f"> Already have '{out_file}', skipping...")
            else:
                self.print(f"> Processing: {processed_file}")
                pybdsf_table = self.__load(processed_file)
                vlad_table = self.__process_modules(pybdsf_table,meta,metrics,processed_file)
                self.print(f"> Writing: {out_file}")
                ascii.write(vlad_table,out_file,format="csv",overwrite=True)
                self.__write_peak_to_ring_reg_file(vlad_table,out_file)
            update()
        if not vlad_table is None:
            self.print(f"> Output catalogue format:")
            self.print(vlad_table.info)
        with ProcessLock() as db:
            db['vlad']['Status'] = "[Done]"
        self.print("[Done]")

        return self

    def compilate(self):
        with ProcessLock() as db:
            if not 'diagnostics' in db.keys():
                db['diagnostics'] = dict()
            db['diagnostics']['vlad_compilate'] = {
                'PID': os.getpid(),
                'Status': "[Running]" 
            }
        dups = self.get_duplicates_checker()
        self.print("Complilating Subtile-based VLAD catalogues")
        vlad_subtile_files = glob.glob(f"{self.get_directories()['tiles']}/*/*.vlad.csv")
        tracker = TimeStats(len(vlad_subtile_files)).start()
        def update():
            s = tracker.update_dt().get_stats()
            s['Status'] = "[Running]"
            with ProcessLock() as db:
                db['diagnostics']['vlad_compilate'] = s
        vlad_catalogue = QTable()
        for vlad_subtile_file in vlad_subtile_files:
            if dups.is_keep(vlad_subtile_file):
                self.print(f"> Stacking: {vlad_subtile_file}")
                try:
                    vlad_catalogue = vstack([vlad_catalogue,QTable.read(vlad_subtile_file)])
                except Exception as e:
                    self.print(f"> CATERR: {e}: SKIPPING: {vlad_subtile_file}")
            else:
                self.print(f"> Older Version: '{vlad_subtile_file}', skipping...")
            update()
        # add units
        for record in self.catalogue['meta']:
            name = list(record.keys())[0]
            if 'units' in record[name].keys():
                for field in vlad_catalogue.colnames:
                    if field == name:
                        units = self.convert_to_astropy_units(record[field]['units'])
                        vlad_catalogue[field] = vlad_catalogue[field] * units
                        break
        with ProcessLock() as db:
            db['diagnostics']['vlad_compilate']['Status'] = "[Done]"
        self.print("[Done]")
        return vlad_catalogue

    def get_vlad_subtile_files(self):
        dups = self.get_duplicates_checker()
        vlad_subtile_files = list()
        my_path = os.path.realpath(__file__)
        for vlad_subtile_file in glob.glob(f"{self.get_directories()['tiles']}/*/*.vlad.csv"):
            if dups.is_keep(vlad_subtile_file):
                vlad_subtile_files.append(vlad_subtile_file)
        return vlad_subtile_files

    def flush(self):
        self.print("Flushing VLAD files...")
        files = glob.glob(f"{self.get_directories()['tiles']}/*/*.vlad.*")
        if len(files) > 0:
            for file in files:
                if os.path.isfile(file):
                    self.print(f"> Flushing: {file}")
                    os.remove(file)
        self.print("[Done]")
        return self


class Diagnostics(VLAD):
    def __init__(self,is_log=True,is_stub_test=False):
        VLAD.__init__(self,is_log)
        self.is_stub_test = is_stub_test
        diagnostics = self.get_diagnostic_settings()
        self.surveys = diagnostics['surveys']
        self.vlad_modules = self.__load_vlad_modules()
        self.vlad_preqa_catalogue_filename = f"{self.get_directories()['diagnostics']}/PreQA_VLAD_Catalogue.csv"
        self.qa_subtile_info_filename = f"{self.get_directories()['diagnostics']}/PreQA_Subtile_Info_Table.csv"
        self.info_table_table = diagnostics['subtile_info_table']

    def get_vlad_csv_filename(self):
        return self.vlad_preqa_catalogue_filename

    def get_diagnostics_table_filename(self):
        return self.qa_subtile_info_filename

    def __load_vlad_modules(self):
        modules = dict()
        for record in self.get_diagnostic_settings()['vlad_modules']:
            name = list(record.keys())[0]
            modules[name] = eval(record[name]['module'])
        return modules

    def compilate_and_update_vlad(self):
        vlad_catalogue = self.compilate()

        self.print("Updating VLAD...")
        # convert vlad catalogue meta list to dict
        meta = dict()
        for record in self.get_vlad_settings()['catalogue']['meta']:
            name = list(record.keys())[0]
            meta[name] = record[name]

        # process vlad modules
        vlad_table = QTable()
        for record in self.get_vlad_settings()['catalogue']['meta']:
            name = list(record.keys())[0]
            if name in self.vlad_modules.keys():
                vlad_table[name] = self.vlad_modules[name](vlad_catalogue,meta,self.surveys)
            elif name in vlad_catalogue.colnames:
                vlad_table[name] = vlad_catalogue[name]

        if len(vlad_table) > 0:
            self.print(f"> Output catalogue format:")
            self.print(vlad_table.info)
            self.print(f"> Dumping catalogue: {self.vlad_preqa_catalogue_filename}")
            ascii.write(vlad_table,self.vlad_preqa_catalogue_filename,format="csv",overwrite=True)
        self.print("[DONE]")
        return self

    def build_subtile_info_table(self):
        with ProcessLock() as db:
            if not 'diagnostics' in db.keys():
                db['diagnostics'] = dict()
            db['diagnostics']['subtile_info_table'] = {
                'PID': os.getpid(),
                'Status': "[Running]" 
            }
        vlad_subtile_files = self.get_vlad_subtile_files()
        tracker = TimeStats(len(vlad_subtile_files)).start()
        def update_callback():
            s = tracker.update_dt().get_stats()
            s['Status'] = "[Running]"
            with ProcessLock() as db:
                db['diagnostics']['subtile_info_table'] = s
        create_table = eval(self.get_diagnostic_settings()['subtile_info_table']['module'])
        urls = QTable.read(self.get_manifest(),format='csv')['file']
        #info_table = create_table(self.get_vlad_subtile_files(),urls)
        info_table = create_table(self.get_vlad_subtile_files(),urls,update_callback)
        if len(info_table) > 0:
            self.print(f"> Dumping catalogue: {self.qa_subtile_info_filename}")
            ascii.write(info_table,self.qa_subtile_info_filename,format="csv",overwrite=True)
            self.print("[Done]")
        with ProcessLock() as db:
            db['diagnostics']['subtile_info_table']['Status'] = "[Done]"
        return self

    def get_info_table_html_help_table(self):
        info_table = self.get_html_help_table(self.get_diagnostic_settings()['subtile_info_table']['meta'])
        return info_table

    def dump_json_info_table_html_help(self):
        json_file = f"{self.get_directories()['products']}/{self.info_table_table['json_help_file']}"
        json_file = re.sub(r"\.json$",f"_{self.get_timestamp()}.json",json_file)
        self.print(f"Dumping Subtile Info Table JSON BDP help meta file: {json_file}")
        with open(json_file,"w+") as f:
            contents = json.dumps(self.get_html_help_dict(self.info_table_table['meta']),indent='   ')
            f.write(contents)
        self.print("[Done]")
        return self

    def dump_html_and_diagnostic_plots(self):
        t_start = time.perf_counter()
        with ProcessLock() as db:
            if not 'diagnostics' in db.keys():
                db['diagnostics'] = dict()
            db['diagnostics']['diagnostic_plots'] = {
                'PID': os.getpid(),
                'Status': "[Running]" 
            }
        if not os.path.isfile(self.qa_subtile_info_filename):
            self.build_subtile_info_table()
        if os.path.isfile(self.qa_subtile_info_filename):
            def wrapper(html_obj):
                html_obj = [
                    f"\n<div style=\"padding-left:10px\">\n",
                    re.sub(r"\n","\n"+(3 * " "),"\n"+html_obj),
                    f"\n</div>\n"
                ]
                return re.sub(r"\n+","\n","\n".join(html_obj)+"\n")
            info_table = pd.read_csv(self.qa_subtile_info_filename)
            plots_dir = self.get_diagnostic_plots_dir()
            html_dir  = self.get_diagnostics_html_dir()
            tracker = TimeStats(len(get_subtile_info_table_plot_params().keys()))
            def update():
                s = tracker.update_dt().get_stats()
                s['Status'] = "[Running]"
                with ProcessLock() as db:
                    db['diagnostics']['diagnostic_plots'] = s
            html_table = create_subtile_info_table_plots(info_table,plots_dir,update,self.is_stub_test)
            html_info_plots = wrapper(re.sub(r"(img\s+src=[\"'])",r"\1../plots/",html_table))
            html_vlad_table = wrapper(self.get_vlad_html_help_table(is_add_div_wrapper=False))
            html_info_table = wrapper(self.get_info_table_html_help_table())
            file_contents = ""
            file_contents += f"<!DOCTYPE html>\n"
            file_contents += f"<html>\n"
            file_contents += f"   <head>\n"
            file_contents += f"      <title>Diagnostics Test HTML File</title>"
            file_contents += f"   </head>\n"
            file_contents += f"   <body>\n"
            file_contents += f"      <h1>VLAD Table Definitions</h1>\n"
            file_contents += re.sub(r"\n\s*$","\n",re.sub(r"\n","\n"+(6 * " "),html_vlad_table))
            file_contents += f"      <h1>Subtile Info. Table Plots and Definitions</h1>\n"
            file_contents += re.sub(r"\n\s*$","\n",re.sub(r"\n","\n"+(6 * " "),html_info_plots))
            file_contents += re.sub(r"\n\s*$","\n",re.sub(r"\n","\n"+(6 * " "),html_info_table))
            file_contents += f"   </body>\n"
            file_contents += f"</html>\n"
            file_contents = re.sub(r"\n+","\n",file_contents)
            html_file = f"{html_dir}/index.html"
            with open(html_file,"w+") as f:
                f.write(file_contents)
        with ProcessLock() as db:
            db['diagnostics']['diagnostic_plots']['Status'] = "[Done]"
                
    def build(self,is_flush=False,is_pull=False,html_only=False):
        with ProcessLock() as db:
            if not 'diagnostics' in db.keys():
                db['diagnostics'] = dict()
            db['diagnostics'] = {
                'diagnostic_plots' : {
                    'PID': os.getpid(),
                    'Status': "[Standby]" 
                },
                'vlad_compilate' : {
                    'PID': os.getpid(),
                    'Status': "[Standby]" 
                },
                'subtile_info_table' : {
                    'PID': os.getpid(),
                    'Status': "[Standby]" 
                },
            }
        if is_flush:
            Diagnostics.flush(self)
        if is_pull:
            VLAD.build(self,is_flush,is_pull)
        self.dump_html_and_diagnostic_plots()
        if not html_only:
            self.compilate_and_update_vlad()
            self.build_subtile_info_table()
        return self

    def flush(self):
        self.print("Flushing diagnostic files...")
        files = glob.glob(f"{self.get_directories()['diagnostics']}/*")
        if len(files) > 0:
            html_files = glob.glob(f"{self.get_diagnostics_html_dir()}/*")
            if len(html_files) > 0:
                files.extend(html_files)
            plot_files = glob.glob(f"{self.get_diagnostic_plots_dir()}/*")
            if len(plot_files) > 0:
                files.extend(plot_files)
            for file in files:
                if os.path.isfile(file):
                    self.print(f"> Flushing: {file}")
                    os.remove(file)
        self.print("[Done]")
        return self


class QA(Diagnostics):
    def __init__(self,is_log=True):
        Diagnostics.__init__(self,is_log)
        qa = self.get_qa_settings()
        self.qa_metrics = qa['metrics']
        self.qa_modules = self.__load_vlad_modules()
        self.diagnostics_vlad_file = self.get_vlad_csv_filename()
        self.diagnostics_info_table_file = self.get_diagnostics_table_filename()

        times_str = self.get_timestamp()
        def add_date(csv_file):
            return re.sub(r"\.csv$",f"_{times_str}.csv",csv_file)
        products_dir = self.get_directories()['products']
        #self.vlad_csv_file         = f"{products_dir}/VLASS1_UOFM_QL_Catalogue_{times_str}.csv"
        self.vlad_csv_file         = add_date(f"{products_dir}/{self.get_vlad_settings()['catalogue']['filename']}")
        #self.subtile_info_csv_file = f"{products_dir}/CIRADA_VLASS1QL_table3_subtile_info_{times_str}.csv"
        self.subtile_info_csv_file = add_date(f"{products_dir}/{self.get_diagnostic_settings()['subtile_info_table']['filename']}")

    def __load_vlad_modules(self):
        modules = dict()
        for record in self.get_qa_settings()['vlad_modules']:
            name = list(record.keys())[0]
            if 'returns' in record[name].keys():
                modules[name] = {
                    'func': eval(record[name]['module']),
                    'returns': record[name]['returns']
                }
            else:
                modules[name] = eval(record[name]['module'])
        #print(modules)
        return modules

    def update_deploy_vlad_table(self):
        if not os.path.isfile(self.diagnostics_vlad_file):
            with ProcessLock() as db:
                db['qa'] = {
                    'PID': os.getpid(),
                    'Status': "[Error]" 
                }
            self.print("> WARNING: PreQA-VLAD '{self.diagnostics_vlad_file}' file not found, can't create VLAD catalogue!")
            return
        with ProcessLock() as db:
            db['qa'] = {
                'PID': os.getpid(),
                'Status': "[Running]" 
            }
        # get the pre-qa vlad catalogue and add units
        vlad_catalogue = QTable.read(self.diagnostics_vlad_file)
        for record in self.get_vlad_settings()['catalogue']['meta']:
            name = list(record.keys())[0]
            if 'units' in record[name].keys():
                for field in vlad_catalogue.colnames:
                    if field == name:
                        units = self.convert_to_astropy_units(record[field]['units'])
                        vlad_catalogue[field] = vlad_catalogue[field] * units
                        break

        # convert vlad meta list to dict
        meta = dict()
        for record in self.get_vlad_settings()['catalogue']['meta']:
            name = list(record.keys())[0]
            meta[name] = record[name]

        # get metrics
        metrics = self.qa_metrics

        # get vlad-qa modules
        modules = self.qa_modules

        # get subtile info
        info_table = QTable.read(self.diagnostics_info_table_file)

        self.print(f"Creating VLAD Catalogue...")
        subtiles = self.get_subtiles()
        tracker = TimeStats(len(modules.keys())).start()
        def update():
            s = {re.sub("Iteration","Module",k): v for k,v in tracker.update_dt().get_stats().items()}
            s['Status'] = "[Running]"
            with ProcessLock() as db:
                db['qa'] = s
            self.print("> - [STATS] QA:\n> - "+"\n> - ".join([f"{re.sub(r'_',' ',k)}: {s[k]}" for k in s]))
        for name in modules.keys():
            self.print(f"> Processing module: {name}")
            if isinstance(modules[name],dict) and 'returns' in modules[name].keys():
                datum = self.qa_modules[name]['func'](vlad_catalogue,meta,self.qa_metrics,info_table)
                for field in datum.keys():
                    vlad_catalogue[field] = datum[field]
            else:
                vlad_catalogue[name] = self.qa_modules[name](vlad_catalogue,meta,self.qa_metrics,info_table)
            update()

        vlad_table = QTable()
        for record in self.get_vlad_settings()['catalogue']['meta']:
            name = list(record.keys())[0]
            if name in vlad_catalogue.keys():
                vlad_table[name] = vlad_catalogue[name]

        if len(vlad_table) > 0:
            self.print(f"> Output catalogue format:")
            self.print(vlad_table.info)
            self.print(f"> Deploying VLAD Catalogue BDP: {self.vlad_csv_file}")
            #ascii.write(vlad_table,self.vlad_csv_file,format="csv",overwrite=True)
            # do some housecleaning.
            df = vlad_table.to_pandas()
            df = df.fillna(-99)
            df = df.sort_values(by='RA', ascending=True).reset_index(drop=True)
            df.to_csv(self.vlad_csv_file,index=False)
        else:
            with ProcessLock() as db:
                db['qa']['Status'] = "[Error]"
            self.print("> WARNING: Empty catalogue.")
        with ProcessLock() as db:
            db['qa']['Status'] = "[Done]"
        self.print("[Done]")
        return self

    def delopy_info_table(self):
        self.print("Deploying Subtile Info. Table...")
        if os.path.isfile(self.diagnostics_info_table_file):
            self.print(f"cp {self.diagnostics_info_table_file} {self.subtile_info_csv_file}")
            copyfile(self.diagnostics_info_table_file,self.subtile_info_csv_file)
        self.print("[Done]")
        return self

    def build(self,is_flush=False,is_pull=False):
        with ProcessLock() as db:
            db['qa'] = {
                'PID': os.getpid(),
                'Status': "[Standby]" if is_pull else "[Running]" 
            }
        if is_flush:
            QA.flush(self)
        if is_pull:
            Diagnostics.build(self,is_flush,is_pull)
        self.update_deploy_vlad_table()
        self.delopy_info_table()
        self.dump_json_vlad_html_help()
        self.dump_json_info_table_html_help()
        return self


    def flush(self):
        self.print("Flushing QA files...")
        files = glob.glob(f"{self.get_directories()['products']}/*")
        if len(files) > 0:
            for file in files:
                if os.path.isfile(file):
                    self.print(f"> Flushing: {file}")
                    os.remove(file)
        self.print("[Done]")
        return self

class Upload(QA):
    def __init__(self,is_log=True):
        QA.__init__(self,is_log)
        upload = self.get_upload_settings()
        self.destination = upload['destination']
        products_dir  = self.get_directories()['products']
        c_vlad        = self.get_vlad_settings()['catalogue']
        c_diagnostics = self.get_diagnostic_settings()['subtile_info_table']
        self.uploads = {
            'vlad': {
                'catalogue': f"{products_dir}/{c_vlad['filename']}",
                'metadata' : f"{products_dir}/{c_vlad['json_help_file']}",
             },
            'info_table': {
                'catalogue': f"{products_dir}/{c_diagnostics['filename']}",
                'metadata' : f"{products_dir}/{c_diagnostics['json_help_file']}",
             },
        }

    def push(self,is_flush=False,is_pull=False):
        with ProcessLock() as db:
            db['upload'] = {
                'PID' : os.getpid(),
                'Status': "[Standby]" if is_pull else "[Running]"
            }
        if is_flush:
            Upload.flush(self)
        if is_pull:
            QA.build(self,is_flush,is_pull)

        self.print(f"Uploading BDPs...")
        max_bdps = 0
        for item in self.uploads.keys():
            max_bdps += len(self.uploads[item].keys())
        tracker = TimeStats(max_bdps).start()
        def update():
            s = tracker.update_dt().get_stats()
            s['Status'] = "[Running]"
            with ProcessLock() as db:
                db['upload'] = s
            self.print("> - [STATS] UPLOAD:\n> - "+f"Elapsed Time: {s['Elapsed_Time']}")
            
        for item in self.uploads.keys():
            for bdp in self.uploads[item]:
                local = np.sort(glob.glob(re.sub(r"\.(csv|json)$",r"_*.\1",self.uploads[item][bdp]))).tolist().pop()
                print(f"> Uploading: {local}")
                remote = f"{self.destination}/{re.sub(r'^(.*?/)*','',local)}"
                print(f"> > $ vcp {local} {remote}")
                VOSpace().copy(local,remote)
                update()
        with ProcessLock() as db:
            db['upload']['Status'] = "[Done]"
        self.print("[Done]")
        return self

    def flush(self):
        # extract remote files related to this pipelines' bdps
        bdps = [re.sub(r"^(.*?/)*","",self.uploads[item][bdp]) for item in self.uploads.keys() for bdp in self.uploads[item]]
        template_bdps = ["^"+re.sub(r"\.(csv|json)$",r"_\\d{8}_\\d{6}\\.\1(|\\.gz)$",bdp) for bdp in bdps]
        def is_bdp(vospace_file):
            for t_bdp in template_bdps:
                if re.search(t_bdp,vospace_file):
                    return True
            return False
        remote_bdps = list()
        for remote_file in VOSpace().listdir(self.destination):
            if is_bdp(remote_file):
                remote_bdps.append(f"{self.destination}/{remote_file}")
        # ok, let's flush them
        self.print("Flushing VOSPace files...")
        for remote_bdp in remote_bdps:
            print(f"> Flushing: {remote_bdp}")
            print(f"> > $ vrm {remote_bdp}")
            VOSpace().delete(remote_bdp)
        self.print("[Done]")
        return self



############################################################
#
#    * * * C O M A N D - L I N E   I N T E R F A C E * * *
#

# need this class to put our help command-line list in pipeline order
# notes: https://stackoverflow.com/questions/47972638/how-can-i-define-the-order-of-click-sub-commands-in-help
class SpecialHelpOrder(click.Group):

    def __init__(self, *args, **kwargs):
        self.help_priorities = {}
        super(SpecialHelpOrder, self).__init__(*args, **kwargs)

    def get_help(self, ctx):
        self.list_commands = self.list_commands_for_help
        return super(SpecialHelpOrder, self).get_help(ctx)

    def list_commands_for_help(self, ctx):
        """reorder the list of commands when listing the help"""
        commands = super(SpecialHelpOrder, self).list_commands(ctx)
        return (c[1] for c in sorted(
            (self.help_priorities.get(command, 1), command)
            for command in commands))

    def command(self, *args, **kwargs):
        """Behaves the same as `click.Group.command()` except capture
        a priority for listing command names in help.
        """
        help_priority = kwargs.pop('help_priority', 1)
        help_priorities = self.help_priorities

        def decorator(f):
            cmd = super(SpecialHelpOrder, self).command(*args, **kwargs)(f)
            help_priorities[cmd.name] = help_priority
            return cmd

        return decorator

# now we can initialize our help command
@click.group(cls=SpecialHelpOrder)
def cli():
    """\b
        VLASS catalogue generator tool.

        This tool is used for building the VLASS CIRADA catalogue Basic
        Data Product (BDP), and related products. The BDPs are generated
        through the following pipeline command line steps:

           \b
           -- configure
           -- download
           -- process
           -- vlad
           -- diagnostics
           -- qa
           -- upload

        Following the command line help in the above order, will serve as
        a small tutorial on configuring and running the pipeline: e.g.,

           \b
           $ python catenator.py configure --help

        The most basic features are --pull and --flush. The --pull flag
        runs every thing upstream of a given command, with the exception
        of configure. The --flush flag will remove files appropriately.
        There is also a --verbose flag for running the pipeline in the
        foreground; otherwise, it will run in the background (recommended).
        
        The progress of the pipeline can be checked anytime
        using the monitor command.
        
        The remaining commands are flush and logger, which are basically
        house keeping commands.
        
        The Quality Assurance (QA) step, qa, should be done after the
        output from the diagnostics step has been carefully examined,
        in order to provide the appropriate QA input metrics for
        calculating QA parameters. The pipeline can be run up to this
        step by entering the following command.

           \b
           $ python catenator.py diagnostics --pull

        Once satisfied, the user can adjust the settings in the configuration,
        create the BDPs and upload them to the CIRADA database: i.e.,

           \b
           $ python catenator.py upload --pull

        CAUTION: Adding the --flush flag would delete everything upstream
        and restart the whole pipeline from the beginning.
    """
    pass

# new we create an auto-priority counter that lists
# the commands in the order they are presented.
class HelpCommandPriority:
    def __init__(self):
        self.counter = 0
    def prio(self):
        self.counter +=1
        return self.counter
cmd=HelpCommandPriority()

@cli.command(help_priority=cmd.prio())
def configure(
):
    """\b
        Select pipeline configuration file.

        The pipeline settings are managed by YAML (.yml) configuration
        files in the ./conf directory. This command allows for the
        selection of those files.

        The following example shows the selection of a test configuration,
        from the current selection (*).

           \b
           $ python catenator.py configure
           Please select configuration no:
           1 pipeline_v1.yml (*)
           2 pipeline_test.yml
           [1-2]/q: 2
           New Configuration: pipeline_test.yml
           $

        A quick check shows the newly selected configuration.

           \b
           $ python catenator.py configure
           Please select configuration no:
           1 pipeline_v1.yml
           2 pipeline_test.yml (*)
           [1-2]/q: q
           Bye!
           $

        New configurations can be added by placing .yml files in the
        configuration directory, under source control. (Also, the 
        desired default configuration, on first checkout of the software,
        can be achieved by selecting the configuration, and then checking
        in the contents of the ./.cache directory.)

        The configuration file is broken down into the following main
        blocks.

           \b
           versioning:
              ...
           directories:
              ...
           downloads:
              ...
           pybdsf:
              ...
           vlad:
              ...
           qa:
              ...
    
        The 'versioning' and 'directories' blocks specify the version
        information and deployment-directories, respectively. The
        remainder specify the configuration in accordance to the pipeline
        steps. Details can be found in the comments of the
        ./config/pipeline_test.yml configuration file, as well as within
        this help utility, for each pipeline step.
    """
    cache = shelve.open(state_cache)
    config_files = [re.sub(r"^(.*?/)*","",f) for f in glob.glob(f"{cache['config_dir']}/*.yml")]
    print(f"Please select configuration no:")
    for i,c in enumerate(config_files):
        print(f"{i+1} {c}{' (*)' if c==cache['config_file'] else ''}")
    while True:
        value = input(f"[1-{len(config_files)}]/q: ")
        if value == 'q':
            break
        if value in [f"{n}" for n in range(1,len(config_files)+1)]:
            value = int(value)-1
            break
    if isinstance(value,int):
        cache['config_file'] = config_files[value]
        print(f"New Configuration: {cache['config_file']}")
    else:
        print("Bye!")

    cache.close()

# global pipeline flag help definitions
pipeline_flag_help_matrix = {
    'download': {
        'flush-all': 'Flush *ALL CONTENTS* of VLASS Tile dirs.',
        'flush': 'Flush VLASS FITS image download files.',
    },
    'process': {
        'pull': 'Do download step first.',
        'flush': 'Flush PyBDSF output .fits.pybdsf.log, .csv, and .reg files.',
    },
    'vlad': {
        'pull': 'Do download and process steps first.',
        'flush': 'Flush VLAD processing .csv and .reg files.',
    },
    'diagnostics': {
        'pull': 'Do download, process, and vlad steps first.',
        'flush': 'Flush diagnostic files.',
    },
    'qa': {
        'pull': 'Do download, process, vlad, and diagnostics steps first.',
        'flush': 'Flush deployed files for upload.',
    },
    'upload': {
        'pull': 'Do download, process, vlad, diagnostics, and qa steps first.',
        'flush': 'Flush VOSpace uploaded files.',
    },
}

#   * * *   D  E P R E C A T E D   * * *
# nohup-like process fork and detached decorator-function
# notes: https://stackoverflow.com/questions/5929107/decorators-with-parameters
def spawn(is_fork=True,message=None):
    def decorator(func):
        def wrapper():
            if is_fork:
                # Do the UNIX double-fork magic [1], see Stevens' "Advanced Programming in the UNIX Environment" for details (ISBN 0201563177).
                # [1] https://stackoverflow.com/questions/6011235/run-a-program-from-python-and-have-it-continue-to-run-after-the-script-is-kille
                try:
                    pid = os.fork()
                    if pid > 0: # parent
                        time.sleep(3)
                        return
                except OSError as e:
                    sys.stderr.write(f"Fork #1 failed: {e.errno} ({e.strerror})")
                    sys.exit(1)
                # child

                # Disconnect from tty (i.e., login session) so process doesn't stop after exiting (i.e., logging out).
                os.setsid() # https://stackoverflow.com/questions/45911705/why-use-os-setsid-in-python

                try:
                    pid = os.fork()
                except OSError as e:
                    sys.stderr.write(f"Fork #w failed: {e.errno} ({e.strerror})")
                    sys.exit(1)

                if pid == 0: # offspring (child 2)
                    func()
                    os._exit(0)
                else: # child (parent 2)
                    time.sleep(1)
                    parent,child = (os.getpid(),pid)
                    msg = list()
                    msg.append(f"FORKING CHILD[{child}]: Check log file and child PID {child} for progress...")
                    if not message is None:
                        msg.append("> ",re.sub(r"\n","\n> ",message))
                    msg.append(f"PARENT [{parent}]: BYE!")
                    print("\n".join(msg))
            else:
                func()
        return wrapper
    return decorator

# nohup-like process fork and detached context-manager 
class PipelineRunning(Exception):
    # raised if pipeline already running
    pass
class spawn2(object):
    def __init__(self,is_fork=True,message=None):
        self.is_fork = is_fork
        self.message = message
        self.child_pid = None
        self.parent_pid = None
        self.process = ProcessLock()

    def __cant_run(self):
        pid = self.process.get_pid()
        if not pid is None:
            msg = list()
            msg.append(f"WARNING: Can't execute command!")
            msg.append(f"Process {pid} is running.")
            msg.append(f"Please wait/kill before starting a new job.")
            print("\n> ".join(msg))
            return True
        return False

    def bail(self,frame,event,arg):
        self.is_fork = False
        raise PipelineRunning()

    def __enter__(self):
        if self.__cant_run():
            # bail using python with hack
            # notes: https://stackoverflow.com/questions/12594148/skipping-execution-of-with-block
            sys.settrace(lambda *args,**keys: None)
            frame = sys._getframe(1)
            frame.f_trace = self.bail
        elif self.is_fork:
            # ok, lets do the UNIX double-fork magic, a la, Stevens' "Advanced
            # Programming in the UNIX Environment" (ISBN 0201563177).

            try: # fork 1
                pid = os.fork()
                if pid > 0: # parent 1
                    time.sleep(3) # so output isn't jumbled.
                    # we need to exit the parent process, so we
                    # don't fall into the with-statement block
                    sys.exit(0)
            except OSError as e:
                sys.stderr.write(f"Fork #1 failed: {e.errno} ({e.strerror})")
                sys.exit(1)
            # child 1

            # Disconnect from tty (i.e., login session) so process doesn't
            # stop after exiting (i.e., logging out): cf., nohup.
            os.setsid()

            try: # fork 2
                pid = os.fork()
                if pid > 0: # parent 2 (child 1)
                    time.sleep(1)
                    self.parent_pid,self.child_pid = (os.getpid(),pid)
                    msg = list()
                    msg.append(f"FORKING CHILD[{self.child_pid}]:")
                    msg.append(f"> Check log file and child PID {self.child_pid} for status.")
                    msg.append(f"> Use the \"{os.path.basename(__file__)} monitor\" command to monitor progress.")
                    if not self.message is None:
                        msg.append("> "+re.sub(r"\n","\n> ",self.message))
                    msg.append(f"PARENT[{self.parent_pid}]: BYE!")
                    print("\n".join(msg))
                    # we need to exit the parent's child process (child 1),
                    # so  we don't fall into the with-statement block
                    os._exit(0)
            except OSError as e:
                sys.stderr.write(f"Fork #2 failed: {e.errno} ({e.strerror})")
                sys.exit(1)
            # child 2: we are in the parent's child's offspring process,
            # and so we want to fall into the with-statement block

            # lock the un/forked process
            self.process.lock(os.getpid())
        else:
            # lock the parent process
            self.process.lock(os.getpid())

        # caution: don't lock outside if-elif-else block...
        # it has unexpected side-effects on self.bail() call.

        return self

    def __exit__(self,exc_type,exc_value,traceback):
        # unlock if current pid is locked
        if os.getpid() == self.process.get_pid():
            self.process.unlock()

        # handle forked state
        if self.is_fork:
            # the child needs to exit gracefully past the with-statement
            # block, so the code can continue executing, so we tidy up
            # when we are done-done...
            def child_exit():
               os._exit(0)
            atexit.register(child_exit)

        # with bypass (self.bail()) trap
        if exc_type is None:
            return None
        elif issubclass(exc_type,PipelineRunning):
            return True


@cli.command(help_priority=cmd.prio())
@click.option('--flush-all', is_flag=True, default=None, help=pipeline_flag_help_matrix['download']['flush-all'])
@click.option('--flush',     is_flag=True, default=None, help=pipeline_flag_help_matrix['download']['flush'])
@click.option('--verbose',   is_flag=True, default=False, help="Verbose output (/w no logging and spawning)")
def download(
    flush_all,
    flush,
    verbose
):
    """\b
        Download VLASS Subtile FITS images.

        This command downloads VLASS FITS image files from a URL CSV
        manifest file [1], specified in the pipeline configuration.

        For example, the location of the test.csv manifest is defined in
        the ./config/pipeline_test.yml test configuration file, as being
        under the  ./media/manifests directory: i.e.,

           \b
           downloads:
               manifest: 'media/manifests/test.csv'

        where the directory is specified relative to *this* command's
        directory. New manifests should be placed under the
        ./media/manifests directory and checked in under source control,
        along with the new configuration.

        In addition, the destination of the downloads is specified in the
        'directories' block of the configuration file: e.g., the test 
        configuration specifies,

           \b
           directories:
               tiles: 'data/tiles'

        In this particular example, the ./data/tiles directory is within
        this command's repo directory, which is ignored (in .gitignore) as
        long as it's under a 'data' parent-directory. (CAUTION: Specifying
        the same directory structures for different configuration
        versions may cause data conflicts.)
         
        The download command will only download files that have not been
        downloaded, unless the --flush / --flush-all option is used (where
        --flush-all supersedes --flush).

        \b
        ---
        [1] The .csv manifest file also accepts file:// prefixes, in 
            addition to http://, https://, etc.
    """
    msg = "\n".join([
        f"This will take a while...",
        f"Expect O(3)-days for full download.",
    ])
    with spawn2(is_fork=not verbose,message=msg):
        if flush_all:
            Downloader(is_log=not verbose).flush(is_all=True).fetch()
        elif flush:
            Downloader(is_log=not verbose).flush().fetch()
        else:
            Downloader(is_log=not verbose).fetch()

@cli.command(help_priority=cmd.prio())
@click.option('--pull',    is_flag=True, default=False, help=pipeline_flag_help_matrix['process']['pull'])
@click.option('--flush',   is_flag=True, default=False, help=pipeline_flag_help_matrix['process']['flush'])
@click.option('--verbose', is_flag=True, default=False, help="Verbose output (/w no logging and spawning)")
def process(
    pull,
    flush,
    verbose
):
    """\b
        Run PyBDSF image processing.

        This command does PyBDSF image processing on the subtiles
        downloaded by the download command.
 
        The PyBDSF pipeline-stage outputs subtile-based log (.fits.pybdsf.log), catalogue
        (.csv), and region (.reg) files into the VLASS Tile download
        directories, as specified in the configuration file: e.g., in
        '.conf/pipeline_test.yml' it's specified as,

          \b
          directories:
              tiles: 'data/tiles'

        The PyBDSF processing parameters are specified in the pipeline
        configuration: re., above example,
        
           \b
           pybdsf:
               processing:
                   rms_box:
                        box_size_pixels:  200
                        step_size_pixels: 50
                   frequency: 2.987741489322E+09
               catalogue:
                   format: 'csv'
                   catalog_type: 'srl'
                   incl_empty: True
               region_file:
                   format: 'ds9'
                   catalog_type: 'srl'
                   incl_empty: True

        where the definitions can be found the PYBDSF online documentation.

        NB: Only format = 'csv' and catalog_type = 'srl' are currently supported.

        The process command will only process files that have not been
        previously processed, unless the --flush option is used. Combining the
        --flush and --pull flags, will flush everything upstream of this pipeline
        step.
    """
    msg = list()
    msg.append(f"This will take a while...")
    if pull:
        msg.append(f"Expect O(2)-weeks for downloading and image processing.")
    else:
        msg.append(f"Expect O(10)-days for full PyBDSF image processing.")
    msg="\n".join(msg)
    with spawn2(is_fork=not verbose,message=msg):
        Process(is_log=not verbose).process(flush,pull)

@cli.command(help_priority=cmd.prio())
@click.option('--pull',    is_flag=True, default=False, help=pipeline_flag_help_matrix['vlad']['pull'])
@click.option('--flush',   is_flag=True, default=False, help=pipeline_flag_help_matrix['vlad']['flush'])
@click.option('--verbose', is_flag=True, default=False, help="Verbose output (/w no logging and spawning)")
def vlad(
    pull,
    flush,
    verbose
):
    """\b
        Creates VLAD Subtile-based catalogues.

        This command creates Value Added (VLAD) subtile-based PyBDSF
        (.vlad.csv) catalogues and Peak_to_ring DS9 region (.reg) files.
        The files are generated from the downloaded FITS images and PyBDSF
        processed files, from the download and process commands, respectively,
        and are output to the VLASS Tile download directories:
        e.g., in '.conf/pipeline_test.yml' it's specified as,

          \b
          directories:
              tiles: 'data/tiles'

        The catalogue layout and processing modules are specified in the
        configuration: re., previous example, the catalogue layout is
        specified in the 'vlad:catalogue:meta:' sub-block, e.g., 
        
           \b
           vlad:
               ...
               catalogue:
                   ...
                   meta:
                       ...
                       - RA:
                             units: 'deg'
                             html_help:
                                 mouseover: 'Right ascension [{units}].'
                       ...
                       - Total_flux:
                             units_old: 'Jy'
                             units: 'mJy'
                             html_help:
                                 mouseover: 'Integrated flux [{units}].'
                       ...

        where, "..., - RA:, ..., - Total_flux:, ...," are specified in
        order of the desired catalogue output columns. The 'units_old'
        specifies the original units of the PyBDSF-based catalogue
        fields, and 'units' specifies the desired / original units: in
        short, if 'units_old' exits it's converted to 'units'; otherwise,
        it's left as 'units'. The accepted unit-strings are defined in
        astropy units. (Fields that have units require these declarations,
        as the information is not persistent in the .csv files. If there
        are no units, then no specification is required.) The mouseover
        over fields are for use with an HTML user interface, which
        provides mouseover tool tips. (Note: the {units} tags, will
        include the 'units' field defined the immediate parent-block.)
        
        The processing modules are python code snippets for creating column
        outputs and are specified in the 'vlad:catalogue:modules:'
        sub-block. For example, the following Python code snippet, in
        ./media/modules/vlad.py, adds an empty islands 'E' code to the
        'S_code' column (re., PyBDSF output),

           \b
           import re
           import numpy as np
           ...
           def s_code(vlad_table,meta,metrics,pybdsf_output_csv_filename):
               s_code = np.array([],dtype=vlad_table['S_Code'].dtype)
               for cid,code in zip(vlad_table['Component_id'],vlad_table['S_Code']):
                   s_code = np.append(s_code,'E' if cid < 0 else code)
               return s_code

        which is exported in './media/modules/__init__.py' as

           \b
           'from .vlad import s_code'

        and defined in the 'vlad:catalogue:modules:' sub-block

          \b
          vlad:
              ...
              catalogue:
                  ...
                  modules:
                      ...
                      - S_Code:
                            module: 's_code'
                      ...

        All modules in this block require the following input arguments,

           \b
            - vlad_table
            - metrics
            - meta
            - pybdsf_output_csv_filename

        where vlad_table is a VLAD .csv catalogue loaded as an astropy
        QTable, metrics is the input parameters (dict()) specified in
        the "vlad:metrics:" sub-block, e.g.,

            \b
            vlad:
                metrics:
                    peak_to_ring_annulus:
                        units: 'arcsec'
                        r_inner: 5
                        r_outer: 10

        which is used by the peak_to_ring module, meta is the
        'vlad:catalogue:meta:' sub-block with catalogue-column dict()
        keys (minus the 'html_help:' sub-block), and
        pybdsf_output_csv_filename is the filename path to the PyBDSF
        subtile-based output catalogue.

        The vlad command will only process files that have not been previously
        processed, unless the --flush option is used. Combining the --flush
        and --pull flags, will flush everything upstream of this pipeline step.
    """
    msg = list()
    msg.append(f"This will take a while...")
    if pull:
        msg.append(f"Expect O(16)-days for downloading, processing, and VLADing.")
    else:
        msg.append(f"Expect O(2)-days for VLADing.")
    msg="\n".join(msg)
    with spawn2(is_fork=not verbose,message=msg):
        VLAD(is_log=not verbose).build(flush,pull)

@cli.command(help_priority=cmd.prio())
@click.option('--pull',      is_flag=True, default=False, help=pipeline_flag_help_matrix['diagnostics']['pull'])
@click.option('--flush',     is_flag=True, default=False, help=pipeline_flag_help_matrix['diagnostics']['flush'])
@click.option('--verbose',   is_flag=True, default=False, help="Verbose output (/w no logging and spawning).")
@click.option('--html-only', is_flag=True, default=False, help="Dump only HTML test-files (/w diagnostics plots).")
@click.option('--stub-test', is_flag=True, default=False, help="*(For testing/debugging purposes.)*")
def diagnostics(
    pull,
    flush,
    verbose,
    html_only,
    stub_test,
):
    """\b
        Create/overwrite diagnostic QA files.

        The purpose of this step is to provide information for the purposes
        of determining the QA parameters, in order to proceed to the next
        pipeline step: i.e., the qa command. The parameters, which are to
        be determined by humans, are defined in pipeline configuration: e.g.,
        in '.conf/pipeline_test.yml' the are defined under the 'qa:' block,
        i.e.,

           \b
           qa:
               metrics:
                   duplicate_flagging:
                       duplicate_search:
                           units: arcsec
                           radius: 2
                   quality_flagging:
                       sig_noise_threshold: 5
                       peak_to_ring_threshold: 2

        The diagnostics command creates a stacked (i.e., compilated) VLAD
        catalogue (named, PreQA_VLAD_Catalogue.csv), a Subtile Info Table
        (named, PreQA_Subtile_Info_Table.csv), diagnostics plots (in a
        'plots' subdir), and a test HTML file (in a 'html' subdir) in the
        diagnostics directory, specified in the pipeline configuration:
        re., above e.g., it's specified as,

           \b
           directories:
               diagnostics: 'data/diagnostics'

        Upon compilation of the VLAD catalogue the FIRST_distance and
        NVSS_distance columns are created (in accordance with the
        'vlad:catalogue:meta:' definitions sub-block), via the
        'diagnostics:vlad_modules:' modules sub-block (see vlad command
        help for details),
        e.g.,

           \b
           diagnostics:
               ...
               vlad_modules:
                   - FIRST_distance:
                         module: 'first_distance'
                   - NVSS_distance:
                         module: 'nvss_distance'

        which takes as function arguments, in the following order, a loaded
        QTable of the compilated VLAD catalogue (vlad_table), the VLAD
        catalogue meta (meta; cf., see vlad command help), and survey
        catalogue information (surveys) defined in the 'diagnostics:surveys:'
        sub-block, i.e.,

           \b
           diagnostics:
               surveys:
                   FIRST:
                       file: 'media/surveys/first_14dec17.fits.gz'
                       header:
                           ra:
                               units: 'deg'
                               field: 'RA'
                           dec:
                               units: 'deg'
                               field: 'DEC'
                   NVSS:
                       file: 'media/surveys/CATALOG41.FIT.gz'
                       header:
                           ra:
                               units: 'deg'
                               field: 'RA(2000)'
                           dec:
                               units: 'deg'
                               field: 'DEC(2000)'

        The Subtile info Table is generated by the Python module,

           \b
           diagnostics:
               ...
               subtile_info_table:
                   module: 'build_subtile_info_table'

        which takes function arguments, in the following order, a list of
        subtile-based .vlad.csv (VLAD) catalogues (vlad_csv_files), and a
        list of VLASS FITS subtile URLs from the download manifest (urls,
        cf., download command). The module returns the table as an astropy
        QTable, in accordance with the catalogue meta (layout) defined in
        the pipeline configuration: i.e.,

            \b
            diagnostics:
                ...
                subtile_info_table:
                    ...
                    meta:
                        - Subtile:
                              html_help:
                                  mouseover: 'Subtile image name.'
                        - Image_version:
                              html_help:
                                  mouseover: 'Subtile version.'
                        ...

        (see vlad command help for details).

        The following diagnostic histograms are generated,

           \b
           - Mean_isl_rms: (Mean Island-RMS)/Subtile (mJy/Beam), 
           - SD_isl_rms: (SD Island-RMS)/Subtile (mJy/Beam), 
           - Peak_flux_p25: (Q1 Peak Flux Density)/Subtile (mJy/Beam), 
           - Peak_flux_p50: (Q2 Peak Flux Density)/Subtile (mJy/Beam), 
           - Peak_flux_p75: (Q3 Peak Flux Density)/Subtile (mJy/Beam), 
           - N_components: Components/Subtile, 
           - N_empty_islands: (Empty Islands)/Subtile, 
           - Peak_flux_max: (Max Peak Flux)/Subtile (mJy/Beam), 

        whose generator scripts can be found in 'media/modules/diagnostic_plots.py'
        module.

        The test HTML file (index.html) contains table definitions information
        for the VLAD catalogue and Subtile Info Table, along with the
        aforementioned diagnostics plots.

        This command will overwrite existing files; the --flush option, will remove
        the files before the command is run. Combining the --flush and --pull flags,
        will flush everything upstream of this pipeline step.

        The --stub-test flag is for producing more realistic plots, during software
        development and testing. This flag should not be used during CIRADA BDPs
        production.
    """
    msg = list()
    msg.append(f"This will take a while...")
    if pull:
        msg.append(f"Expect O(18)-days for downloading, processing, VLADing, diagnostics.")
    else:
        msg.append(f"Expect O(2)-days for diagnostics.")
    msg="\n".join(msg)
    with spawn2(is_fork=not verbose,message=msg):
        Diagnostics(is_log=not verbose,is_stub_test=stub_test).build(flush,pull,html_only)

@cli.command(help_priority=cmd.prio())
@click.option('--pull',    is_flag=True, default=False, help=pipeline_flag_help_matrix['qa']['pull'])
@click.option('--flush',   is_flag=True, default=False, help=pipeline_flag_help_matrix['qa']['flush'])
@click.option('--verbose', is_flag=True, default=False, help="Verbose output (/w no logging and spawning)")
def qa(
    pull,
    flush,
    verbose
):
    """\b
        Adds QA flags to VLAD and deploys BDPs.

        This command adds the Quality Assurance (QA) parameters to the VLAD
        catalogue, and then places the following Basic Data Products (BDPs)
        into a products directory for upload:

           \b
           -- VLASS VLAD Catalogue (.csv)
           -- VLASS VLAD Catalogue HTML Help Meta (.json)
           -- Subtile Info Table (.csv)
           -- Subtile Info Table HTML Help Meta (.json)

        The filename structures are defined in the pipeline
        configuration: e.g., in '.conf/pipeline_test.yml', they are
        defined as

           \b
           vlad:
               ...
               catalogue:
                   filename: 'VLASS1_UOFM_QL_Catalogue.csv'
                   json_help_file: 'VLASS_Component_Catalogue_Help_Map.json'
               ...
           diagnostics:
               ...
               subtile_info_table:
                   filename: 'CIRADA_VLASS1QL_table3_subtile_info.csv'
                   json_help_file: 'VLASS_Subtile_Info_Table_Help_Map.json'

        respectively, and are written to the products directory, defined by,

           \b
           directories:
               products: 'data/products'

        which are suffixed with a unique __YYYYMMDD_HHMMSS.{csv|json}
        timestamp, upon creation, e.g.,

           \b
           CIRADA_VLASS1QL_table3_subtile_info_20200604_225732.csv

        The QA parameters are added to the VLAD catalogue, via, the 
        'qa:vlad_modules' sub-block, re., e.g.,

           \b
           qa:
               ...
               vlad_modules:
                   - xy_positions:
                         module: xy_positions
                         returns:
                             - Xposn
                             - E_Xposn
                             - Yposn
                             - E_Yposn
                             - Xposn_max
                             - E_Xposn_max
                             - Yposn_max
                             - E_Yposn_max
                   ...

        where 'xy_positions' is defined in 'media/modules/qa.py' as
        xy_positions(vlad_table, meta, metrics, info_table), where
        vlad_table is a compilated VLAD catalogue loaded as a QTable,
        meta is the VLAD catalogue template loaded as a dict (minus the
        html_help sub-blocks) from the 'vlad:catalogue:modules:meta:'
        sub-block, metrics are the QA input metrics defined in the
        configuration 'qa:' block, e.g.,

           \b
           qa:
               metrics:
                   ...
                   quality_flagging:
                       sig_noise_threshold: 5
                       peak_to_ring_threshold: 2

        and info_table is the Subtile Info Table loaded as a QTable
        (note, every QA module call receives these arguments): cf.,
        vlad and diagnostics command help for more details.

        The JSON files contain the correspond catalogue html_help meta
        in JSON format for tool-tips (i.e., the 'mouseover' fields) and
        definition tables (i.e., the 'mouseover' + 'expanded' fields).

        This command will overwrite existing files; the --flush option,
        will remove the files before the command is run. Combining the
        --flush and --pull flags, will flush everything upstream of
        this pipeline step.
    """
    msg = list()
    if pull:
        msg.append(f"This will take a while...")
        msg.append(f"Expect O(19)-days for full pipeline run.")
    else:
        msg.append(f"This will take few ticks...")
        msg.append(f"Expect O(6)-hours for QA.")
    msg="\n".join(msg)
    with spawn2(is_fork=not verbose,message=msg):
        QA(is_log=not verbose).build(flush,pull)

@cli.command(help_priority=cmd.prio())
@click.option('--pull',    is_flag=True, default=False, help=pipeline_flag_help_matrix['upload']['pull'])
@click.option('--flush',   is_flag=True, default=False, help=pipeline_flag_help_matrix['upload']['flush'])
@click.option('--verbose', is_flag=True, default=False, help="Verbose output (/w no logging and spawning)")
def upload(
    pull,
    flush,
    verbose
):
    """\b
        Upload BDPs to CIRADA Database.

        This command uploads the BDPs, generated by the qa command, to
        the CANFAR VOSpace storage area, specified in the pipeline
        configuration: e.g., in '.conf/pipeline_test.yml', the location
        is specified in the 'upload:destination' sub-block as

           \b
           upload:
               destination: vos:/cirada/continuum/mboyce/pipeline_upload_test

        It should be noted, that one requires an account in order to 
        access this area: i.e., via the getCert command; e.g.,

           \b
           $ getCert
           www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca Username: bsimpson
           bsimpson@www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca
           Password:
        
           $

        If you see the following error,

           \b
           SSL: SSLV3_ALERT_CERTIFICATE_EXPIRED

        the getCert command will need to be run again, to reissued a new
        certificate. For more details please visit

           \b
           https://www.canfar.net/en/docs/storage/

        This command will upload the qa-command-generated timestamped files
        (BDPs), in an overwrite manner; the --flush option, will remove the
        remote-VOSpace files before the command is run. Combining the --flush
        and --pull flags, will flush everything upstream of this pipeline step.
    """
    msg = list()
    if pull:
        msg.append(f"This will take a while...")
        msg.append(f"Expect O(19)-days for full pipeline run.")
    else:
        msg.append(f"This will take few ticks...")
    msg="\n".join(msg)
    with spawn2(is_fork=not verbose,message=msg):
        Upload(is_log=not verbose).push(flush,pull)

@cli.command(help_priority=cmd.prio())
@click.option('--tile-dirs',        is_flag=True, default=None, help=pipeline_flag_help_matrix['download']['flush-all'])
@click.option('--download-files',   is_flag=True, default=None, help=pipeline_flag_help_matrix['download']['flush'])
@click.option('--process-files',    is_flag=True, default=None, help=pipeline_flag_help_matrix['process']['flush'])
@click.option('--vlad-files',       is_flag=True, default=None, help=pipeline_flag_help_matrix['vlad']['flush'])
@click.option('--diagnostic-files', is_flag=True, default=None, help=pipeline_flag_help_matrix['diagnostics']['flush'])
@click.option('--qa-files',         is_flag=True, default=None, help=pipeline_flag_help_matrix['qa']['flush'])
@click.option('--upload-files',     is_flag=True, default=None, help=pipeline_flag_help_matrix['upload']['flush'])
def flush(
    tile_dirs,
    download_files,
    process_files,
    vlad_files,
    diagnostic_files,
    qa_files,
    upload_files
):
    """\b
        Pipeline file cleanup utility.
    """
    if tile_dirs:
        print("   ***<Tile Dirs>***")
        Downloader(is_log=False).flush(is_all=True)

    if download_files:
        print("   ***<Tile Dirs>***")
        Downloader(is_log=False).flush(is_all=False)

    if process_files:
        print("   ***<Image Processing Files>***")
        Process(is_log=False).flush()

    if vlad_files:
        print("   ***<VLAD Processing Files>***")
        VLAD(is_log=False).flush()

    if diagnostic_files:
        print("   ***<Diagnostic Processing Files>***")
        Diagnostics(is_log=False).flush()

    if qa_files:
        # TO-DO: Need a prune feature
        print("   ***<QA Processing Files>***")
        QA(is_log=False).flush()

    if upload_files:
        # TO-DO: Need a prune feature
        print("   ***<Upload Processing Files>***")
        Upload(is_log=False).flush()
 
@cli.command(help_priority=cmd.prio())
@click.option('--flush-all-logs',         is_flag=True, default=None, help='Flush *ALL* log files.')
@click.option('--flush-download-logs',    is_flag=True, default=None, help='Flush FITS image download log files.')
@click.option('--flush-process-logs',     is_flag=True, default=None, help='Flush PyBDSF image processing log files.')
@click.option('--flush-vlad-logs' ,       is_flag=True, default=None, help='Flush VLAD processing log files.')
@click.option('--flush-diagnostics-logs', is_flag=True, default=None, help='Flush Diagnostic processing log files.')
@click.option('--flush-qa-logs',          is_flag=True, default=None, help='Flush QA processing log files.')
@click.option('--prune-all-logs',         is_flag=True, default=None, help='Prune *ALL* log files.')
@click.option('--prune-download-logs',    is_flag=True, default=None, help='Prune FITS image download log files.')
@click.option('--prune-process-logs',     is_flag=True, default=None, help='Prune PyBDSF image processing log files.')
@click.option('--prune-vlad-logs',        is_flag=True, default=None, help='Prune VLAD processing log files.')
@click.option('--prune-diagnostics-logs', is_flag=True, default=None, help='Prune Diagnostic processing log files.')
@click.option('--prune-qa-logs',          is_flag=True, default=None, help='Prune QA processing log files.')
def logger(
    flush_all_logs,
    flush_download_logs,
    flush_process_logs,
    flush_vlad_logs,
    flush_diagnostics_logs,
    flush_qa_logs,
    prune_all_logs,
    prune_download_logs,
    prune_process_logs,
    prune_vlad_logs,
    prune_diagnostics_logs,
    prune_qa_logs
):
    """\b
        Log file management utility.
    """
    if flush_all_logs:
            print("   ***<Flush *ALL* Logs>***")
            Downloader(is_log=False).flush_logs(is_prune=False)
            Process(is_log=False).flush_logs(is_prune=False)
            VLAD(is_log=False).flush_logs(is_prune=False)
            Diagnostics(is_log=False).flush_logs(is_prune=False)
            QA(is_log=False).flush_logs(is_prune=False)
    elif prune_all_logs:
            print("   ***<Prune *ALL* Logs>***")
            Downloader(is_log=False).flush_logs(is_prune=True)
            Process(is_log=False).flush_logs(is_prune=True)
            VLAD(is_log=False).flush_logs(is_prune=True)
            Diagnostics(is_log=False).flush_logs(is_prune=True)
            QA(is_log=False).flush_logs(is_prune=True)
    else:
        if flush_download_logs:
            print("   ***<Flush Download Logs>***")
            Downloader(is_log=False).flush_logs(is_prune=False)
        elif prune_download_logs:
            print("   ***<Prune Download Logs>***")
            Downloader(is_log=False).flush_logs(is_prune=True)

        if flush_process_logs:
            print("   ***<Flush Process Logs>***")
            Process(is_log=False).flush_logs(is_prune=False)
        elif prune_process_logs:
            print("   ***<Prune Process Logs>***")
            Process(is_log=False).flush_logs(is_prune=True)

        if flush_vlad_logs:
            print("   ***<Flush VLAD Logs>***")
            VLAD(is_log=False).flush_logs(is_prune=False)
        elif prune_vlad_logs:
            print("   ***<Prune VLAD Logs>***")
            VLAD(is_log=False).flush_logs(is_prune=True)

        if flush_diagnostics_logs:
            print("   ***<Flush Diagnostics Logs>***")
            Diagnostics(is_log=False).flush_logs(is_prune=False)
        elif prune_diagnostics_logs:
            print("   ***<Prune Diagnostics Logs>***")
            Diagnostics(is_log=False).flush_logs(is_prune=True)

        if flush_qa_logs:
            print("   ***<Flush QA Logs>***")
            QA(is_log=False).flush_logs(is_prune=False)
        elif prune_qa_logs:
            print("   ***<Prune QA Logs>***")
            QA(is_log=False).flush_logs(is_prune=True)

# pipeline monitoring helper-class
class PipelineMonitor:
    def __init__(self):
        self.pipeline_states = pipeline_states
        self.manager = enlighten.get_manager()
        self.max_field_size = self.__get_pbars_max_descriptor_field_size()
        self.pbars = None

    def __get_pbars_max_descriptor_field_size(self):
        max_field_size = 0
        for state in self.pipeline_states.keys():
            item = self.pipeline_states[state]
            if isinstance(self.pipeline_states[state],dict):
                for sub_state in self.pipeline_states[state].keys():
                    if len(sub_state) > max_field_size:
                        max_field_size = len(sub_state)
            elif len(state) > max_field_size:
                max_field_size = len(state)
        return max_field_size

    def get_processing_stack(self,is_print=True):
        time.sleep(1)
        stack = list()
        with ProcessLock() as db:
            for state in self.pipeline_states.keys():
                if state in db.keys():
                    contents = self.pipeline_states[state]
                    if isinstance(contents,dict):
                        for sub_state in db[state].keys():
                            stack.append(sub_state)
                    else:
                        stack.append(state)
        if is_print and len(stack) > 0:
            print(f"{self.__desc('processing_stack')} {stack}")
        return stack

    def __desc(self,field,is_left_justified=False):
        desc = f"{field}:"
        if is_left_justified:
            return desc + (" " * (self.max_field_size-len(desc)+1))
        return (" " * (self.max_field_size-len(desc)+1)) + desc 

    def __get_states(self):
        states = dict()
        with ProcessLock() as db:
            for state in self.pipeline_states.keys():
                if state in db.keys():
                    contents = self.pipeline_states[state]
                    if isinstance(contents,dict):
                        for sub_state in db[state].keys():
                            n_0 = None
                            n_max = None
                            for item in db[state][sub_state].keys():
                                if item in ['Iteration','Module']:
                                    n_0 = db[state][sub_state][item]
                                elif item in ['Max_Iterations','Max_Modules']:
                                    n_max = db[state][sub_state][item]
                            if not n_0 is None and not n_max is None:
                                if not state in states.keys():
                                    states[state] = dict()
                                states[state][sub_state] = {'n': n_0, 'n_max': n_max}
                    else:
                        n_0 = None
                        n_max = None
                        for item in db[state].keys():
                            if item in ['Iteration','Module']:
                                n_0 = db[state][item]
                            elif item in ['Max_Iterations','Max_Modules']:
                                n_max = db[state][item]
                        if not n_0 is None and not n_max is None:
                            states[state] = {'n': n_0, 'n_max': n_max}
        return states

    def __init_pbars(self):
        states = self.__get_states()
        pbars = dict()
        for state in states.keys():
            pbars[state] = dict()
            if 'n_max' in states[state].keys():
                n     = states[state]['n']
                n_max = states[state]['n_max']
                pbars[state] = {'n': n, 'pbar': self.manager.counter(total=n_max, desc=self.__desc(state))}
                pbars[state]['pbar'].update(n)
            else:
                for sub_state in states[state].keys():
                    n     = states[state][sub_state]['n']
                    n_max = states[state][sub_state]['n_max']
                    pbars[state][sub_state] = {'n': n, 'pbar': self.manager.counter(total=n_max, desc=self.__desc(sub_state))}
                    pbars[state][sub_state]['pbar'].update(n)
        return pbars

    def get_status(self,is_print=True):
        pid = ProcessLock().get_pid()
        if not pid is None:
            if len(self.get_log(False)) > 0:
                msg = f"   * * * PIPELINE PID {pid} IS RUNNING IN BACKGROUND * * *"
            else:
                msg = f"   * * * PIPELINE PID {pid} IS RUNNING IN CONSOLE * * *"
        else:
            msg = f"   * * * PIPELINE IS NOT RUNNING * * *"
        if is_print:
            print(msg)
        return msg

    def get_states(self,is_print=True):
        msg = list()
        with ProcessLock() as db:
            for state in self.pipeline_states.keys():
                if state in db.keys():
                    contents = self.pipeline_states[state]
                    if isinstance(contents,dict):
                        msg.append(f"{state.upper()}: {contents[state] if state in contents.keys() else ''}")
                        for sub_state in db[state].keys():
                            msg.append(f"> {sub_state.upper()}: {contents[sub_state]}")
                            for item in db[state][sub_state].keys():
                                msg.append(f"> > {re.sub(r'_',' ',item)}: {db[state][sub_state][item]}")
                    else:
                        msg.append(f"{state.upper()}: {self.pipeline_states[state]}")
                        for item in db[state].keys():
                            msg.append(f"> {re.sub(r'_',' ',item)}: {db[state][item]}")
        msg = "\n".join(msg)
        if is_print and len(msg) > 0:
            print(msg)
        return msg

    def get_log(self,is_print=True):
        msg = list()
        with ProcessLock() as db:
            if 'log_file' in db.keys():
                if os.path.isfile(db['log_file']):
                   msg.append(f"LOG_FILE: {db['log_file']}")
                else:
                    del db['log_file']
        msg = "\n".join(msg)
        if is_print and len(msg) > 0:
                print(msg)
        return msg

    def get_stack(self,is_print=True):
        msg = list()
        msg.append(self.get_status(False))
        msg.append(self.get_states(False))
        msg.append(self.get_status(False))
        msg.append(self.get_log(False))

        msg = "\n".join(msg)
        if is_print and len(msg) > 0:
            print(msg)
        return msg

    def update_pbars(self):
        if self.pbars is None:
            self.pbars = self.__init_pbars()
        is_new = False
        states = self.__get_states()
        for state in states.keys():
            if 'n' in states[state].keys():
                n = states[state]['n']
                if state in self.pbars.keys():
                    dn = n - self.pbars[state]['n']
                    self.pbars[state]['pbar'].update(dn)
                    self.pbars[state]['n'] = n
                else:
                    n_max = states[state]['n_max']
                    self.pbars[state] = {'n': n, 'pbar': self.manager.counter(total=n_max, desc=self.__desc(state))}
                    self.pbars[state]['pbar'].update(n)
                    is_new = True
            else:
                for sub_state in states[state].keys():
                    n = states[state][sub_state]['n']
                    if state in self.pbars.keys() and sub_state in self.pbars[state].keys():
                        dn = n - self.pbars[state][sub_state]['n']
                        self.pbars[state][sub_state]['pbar'].update(dn)
                        self.pbars[state][sub_state]['n'] = n
                    else:
                        if not state in self.pbars.keys():
                            self.pbars[state] = dict()
                        n_max = states[state][sub_state]['n_max']
                        self.pbars[state][sub_state] = {'n': n, 'pbar': self.manager.counter(total=n_max, desc=self.__desc(sub_state))}
                        self.pbars[state][sub_state]['pbar'].update(n)
                        is_new = True
        return is_new

    def pbars_off(self):
        self.manager.stop()
        self.pbars = None
        self.manager = enlighten.get_manager()


# pipeline monitoring
@cli.command(help_priority=cmd.prio())
@click.option('--continuous', is_flag=True, default=False, help="Show progress bar/s.")
def monitor(
    continuous
):
    """\b
        Pipeline runtime monitoring utility.
    """

    p = PipelineMonitor()
    if not continuous:
        p.get_stack()
    else:
        p.get_status()
        p.get_log()
        p.get_processing_stack()
        while ProcessLock().is_locked():
            time.sleep(1)
            p.update_pbars()
        for _ in range(3):
            time.sleep(1)
            p.update_pbars()
        p.pbars_off()
        print()
        p.get_status()


if __name__ == "__main__":
    cli()


