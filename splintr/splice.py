from splintr.util import vprint
import numpy as np
import pandas as pd
import pickle
from tqdm.autonotebook import tqdm
tqdm.pandas()
import swifter

def rmats_subset_top_events(rmats_data, num_classes):
    '''
    Given RMATS file, subset to top `num_classes` classes of events.
    '''
    jc = pd.read_csv(rmats_data, sep='\t')
    
    is_bg = (jc['sample'] == 'bg')
    bg = jc.loc[is_bg]
    jc = jc.loc[~is_bg]
    class_counts = jc['sample'].value_counts()
    jc_top = jc.loc[jc['sample'].isin(class_counts.index[:num_classes])]
    jc_top = pd.concat([jc_top, bg], ignore_index=True)
    return jc_top
    

def concat(datasets):
    '''
    Combine SpliceData datasets into a single SpliceData object.
    
    datasets ([SpliceData]) : list of SpliceData objects
    '''
    all_events = []
    for d in datasets:
        all_events.extend(d.events)
    return SpliceData(all_events)


class SpliceData():
    def __init__(self, rmats_data):
        '''
        rmats_data (str, pd.DataFrame, [SpliceEvent]) : accepts either a path to the RMATS junction count file, an equivalent DataFrame, or list of SpliceEvent objects. Expects last two columns to be "event" and "sample."
        
        event_type (str) : SE, RI, A5SS, A3SS, or MXE
            
        '''
        vprint('Initializing SpliceData object...')
        if type(rmats_data) is str:
            data = pd.read_csv(rmats_data, sep='\t')
        elif type(rmats_data) is pd.DataFrame:
            data = rmats_data
        elif type(rmats_data) is list:
            self.events = rmats_data
            return
        else:
            raise Exception('Data type not recognized. rmats_data must be a str, DataFrame or list.')
        
        vprint('| Creating events...')
        self.events = data.swifter.apply(SpliceEvent, axis=1).tolist()
        
        vprint('| Done.')
        
    def get_junction_regions(self, exon_in, intron_in):
        '''
        Returns interval around junction. Distance to read into feature specified by exon_in and intron_in.
        
        Parameters
        ----------
        exon_in, intron_in (int) : distance to read into exon and intron respectively
        '''
        regions = []
        pbar = tqdm(self.events, leave=False)
        for event in pbar:
            regions.append(event.get_junction_regions(exon_in, intron_in))
        pbar.close()
        
        return regions
    
    def save(self, file_prefix):
        pickle.dump(self, open(f'{file_prefix}.p', "wb"))

    def load(file_name):
        return pickle.load(open(file_name, 'rb'))
    

class SpliceEvent():
    '''
    This class represents a splice event as the set of involved exons (and introns in the case of Retained Intron events).
    '''
    def __init__(self, event):
        '''
        Parameters
        ----------
        event (pd.Series, dict) : a row from an RMATS junction counts file
                
        Attributes
        ----------
        event_type (str) : SE, RI, A5SS, A3SS, or MXE; inferred from event column
        
        sample (str) : sample name inferred from sample column
        
        gene_id (str) : Ensembl gene ID
        
        gene (str) : Gene symbol
        
        chrom (str) : chromosome
        
        strand (str) : + or -
        
        pvalue (float) : significance of alternatively spliced event
        
        fdr (float) : FDR-corrected p-value
        
        psi ([float]) : percent spliced in of case sample; value for each replicate
        
        control_psi ([float]) : percent spliced in of control; value for each replicate
        
        dpsi (float) : psi - control_psi (average difference for replicates)
        
        alt_exons ([Exon]) : list of the alternatively spliced exon(s) (intron for RI events)
            List should always be sorted from 5' to 3'.
            
        const_exons ([Exon]) : list of constitutively exons
            List should always be sorted from 5' to 3'.
            
        junction_regions ([])
        '''
        # Set attributes
        self.event_type = event.event
        self.sample = event['sample']
        self.gene_id = event.GeneID
        self.gene = event.geneSymbol
        self.chrom = event.chr
        self.strand = event.strand
        self.pvalue = float(event.PValue)
        self.fdr = float(event.FDR)
        self.psi = [float(psi) if psi != 'NA' else float(0) for psi in event.IncLevel1.split(',')]
        self.control_psi = [float(psi) if psi != 'NA' else float(0) for psi in event.IncLevel2.split(',')]
        self.dpsi = float(event.IncLevelDifference) if (event.IncLevelDifference is not 'NA') else float(0)
        
        # Set exons
        if self.event_type == 'SE':
            self.alt_exons = [Exon(event.chr, event.exonStart_0base, event.exonEnd)]
            self.const_exons = [Exon(event.chr, event.upstreamES, event.upstreamEE),
                                Exon(event.chr, event.downstreamES, event.downstreamEE)]
        elif self.event_type == 'RI':
            self.alt_exons = [Exon(event.chr, event.upstreamEE, event.downstreamES)]
            self.const_exons = [Exon(event.chr, event.upstreamES, event.upstreamEE),
                                Exon(event.chr, event.downstreamES, event.downstreamEE)]
        elif self.event_type == 'A5SS':
            self.alt_exons = [Exon(event.chr, event.shortES, event.shortEE)]
            self.const_exons = [Exon(event.chr, event.flankingES, event.flankingEE),
                                Exon(event.chr, event.shortEE, event.longExonEnd)]
        elif self.event_type == 'A3SS':
            self.alt_exons = [Exon(event.chr, event.shortES, event.shortEE)]
            self.const_exons = [Exon(event.chr, event.longExonStart_0base, event.shortES),
                                Exon(event.chr, event.flankingES, event.flankingEE)]
        elif self.event_type == 'MXE':
            self.alt_exons = [Exon(event.chr, event['1stExonStart_0base'], event['1stExonEnd']),
                              Exon(event.chr, event['2ndExonStart_0base'], event['2ndExonEnd'])]
            self.const_exons = [Exon(event.chr, event.upstreamES, event.upstreamEE),
                                Exon(event.chr, event.downstreamES, event.downstreamEE)]
        else:
            raise Exception(f'Event type "{event_type}" not recognized. The SpliceEvent object was not created.')

        self.junction_regions = None
    
    def get_junction_regions(self, exon_in, intron_in):
        '''
        Gets intervals around splice junctions.
        In the case of adjacent "exons" (e.g. intron retention intron is treated as an exon),
        upstream the junction is returned as region and downstream the junction is returned as another.
        
        Parameters
        ----------
        exon_in, intron_in (int) : distance to read into exon and intron respectively
        
        '''
        up = self.const_exons[0]
        down = self.const_exons[1]
            
        exons_sorted = [up]
        exons_sorted.extend(self.alt_exons)
        exons_sorted.append(down)
        
        regions = []
        for i, exon in enumerate(exons_sorted):
            if i == len(exons_sorted) - 1:
                break
            
            regions.append(self._get_3p_region(exons_sorted[i], exons_sorted[i + 1], exon_in, intron_in))
            regions.append(self._get_5p_region(exons_sorted[i], exons_sorted[i + 1], exon_in, intron_in))
        
        for r in regions:
            r.insert(0, self.chrom)
        
        self.junction_regions = regions
        return regions
        
    def _get_3p_region(self, up_exon, down_exon, exon_in, intron_in):
        '''
        Given a pair of exons, get interval around 3p end of upstream exon.
        
        Parameters
        ----------
        exon_in, intron_in (int) : distance to read into exon and intron respectively
        '''
        start = max(up_exon.start, up_exon.end - exon_in)
        end = min(down_exon.start, up_exon.end + intron_in)
        
        assert start < end
        return [start, end]

    def _get_5p_region(self, up_exon, down_exon, exon_in, intron_in):
        '''
        given a pair of exons, get interval around 5p end of downstream exon.
        
        Parameters
        ----------
        exon_in, intron_in (int) : distance to read into exon and intron respectively
        '''
        start = max(up_exon.end, down_exon.start - intron_in)
        end = min(down_exon.end, down_exon.start + exon_in)

        assert start < end
        return [start, end]

class Exon():
    '''
    Simple representation of exon as its coordinates.
    '''
    def __init__(self, chrom, start, end):
        '''
        chrom (str) : chromosome number
        start (int) : start coordinate
        end (int) : end coordinate
        '''
        if not chrom.startswith('chr'):
            raise Exception('chrom must be formatted chrX')
        self.chrom = chrom
        self.start = int(start)
        self.end = int(end)