import os
import csv
import array
import base64
import xmltodict
import re
import numpy as np

class ECGXMLReader:

    def __init__(self, path, augmentLeads=False):
      
            with open(path, 'rb') as xml:
                self.ECG = xmltodict.parse(xml.read().decode('utf8'))
            
            self.augmentLeads           = augmentLeads
            self.path                   = path

            #self.PatientDemographics    = self.ECG['RestingECG']['PatientDemographics']
            #self.TestDemographics       = self.ECG['RestingECG']['TestDemographics']
            self.RestingECGMeasurements = self.ECG['RestingECG']['RestingECGMeasurements']
            #self.MedianBeat              = self.ECG['RestingECG']['Waveform'][0]
            self.Waveforms              = self.ECG['RestingECG']['Waveform'][1]
            self.Median                 = self.ECG['RestingECG']['Waveform'][0]
            self.LeadVoltages           = self.makeLeadVoltages()
            self.MedianVoltages         = self.makeMedianVoltages()
    
    def makeMedianVoltages(self):

        num_leads = 0

        leads = {}

        for lead in self.Median['LeadData']:
            num_leads += 1
            
            lead_data = lead['WaveFormData']
            lead_waveform_data = lead_data['#text']
            #lead_data_clean = re.sub('[^A-Za-z0-9]+', '', lead_data)
            lead_b64  = base64.b64decode(lead_waveform_data)
            lead_vals = np.array(array.array('h', lead_b64))

            leads[lead['LeadID']['#text']] = lead_vals
        return leads   


    def makeLeadVoltages(self):

        num_leads = 0

        leads = {}
        leadsMedianBeat = {}

        for lead in self.Waveforms['LeadData']:
            num_leads += 1
            
            lead_data = lead['WaveFormData']
            lead_waveform_data = lead_data['#text']
            #lead_data_clean = re.sub('[^A-Za-z0-9]+', '', lead_data)
            lead_b64  = base64.b64decode(lead_waveform_data)
            lead_vals = np.array(array.array('h', lead_b64))

            leads[lead['LeadID']['#text']] = lead_vals
        
        # for lead in self.MedianBeat['LeadData']:
        #     num_leads += 1
            
        #     lead_data = lead['WaveFormData']
        #     lead_waveform_data = lead_data['#text']
        #     #lead_data_clean = re.sub('[^A-Za-z0-9]+', '', lead_data)
        #     lead_b64  = base64.b64decode(lead_waveform_data)
        #     lead_vals = np.array(array.array('h', lead_b64))
        #     leadsMedianBeat[leadMedianBeat['LeadID']['#text']] = lead_vals
        
        if num_leads == 8 and self.augmentLeads:

            leads['III'] = np.subtract(leads['II'], leads['I'])
            leads['aVR'] = np.add(leads['I'], leads['II'])*(-0.5)
            leads['aVL'] = np.subtract(leads['I'], 0.5*leads['II'])
            leads['aVF'] = np.subtract(leads['II'], 0.5*leads['I'])
        
        return leads

    def getLeadVoltages(self, LeadID):
        return self.LeadVoltages[LeadID]
    
    def getAllVoltages(self):
        return self.LeadVoltages

    def getAllMedianVoltages(self):
        return self.MedianVoltages