import numpy as np
import pandas as pd
import smtplib
import os
import time as tm
from datetime import datetime
from configparser import ConfigParser
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg') #non-interactive backends for png files
from matplotlib.ticker import MaxNLocator
import torch

class Util:
    def __init__(self, model_descr, dataset_type='notebooks', version=0, prefix=''):
        current_time = datetime.now()
        self.model_descr = model_descr
        self.start_time = current_time.strftime('%d/%m/%Y %H:%M:%S')
        self.start_time_timestamp = tm.time()
        self.version = str(version)
        prefix = prefix.lower() + '_' if prefix.strip() else ''
        self.base_filename =  prefix + self.version + '_' + current_time.strftime('%Y%m%d-%H%M%S')
        self.project_dir = str(Path(__file__).absolute().parent.parent)
        self.output_dir = os.path.join(self.project_dir, 'output', dataset_type)
        
    def plot(self, data, columns_name, x_label, y_label, title, enable=True, inline=False):
        if (enable):
            df = pd.DataFrame(data).T    
            df.columns = columns_name
            df.index += 1
            plot = df.plot(linewidth=2, figsize=(15,8), color=['darkgreen', 'orange'], grid=True);
            train = columns_name[0]
            val = columns_name[1]
            # find position of lowest validation loss
            idx_min_loss = df[val].idxmin()
            plot.axvline(idx_min_loss, linestyle='--', color='r',label='Best epoch');
            plot.legend();
            plot.set_xlim(0, len(df.index)+1);
            plot.xaxis.set_major_locator(MaxNLocator(integer=True))
            plot.set_xlabel(x_label, fontsize=12);
            plot.set_ylabel(y_label, fontsize=12);
            plot.set_title(title, fontsize=16);
            if (not inline):
                plot_dir = self.__create_dir('plots')
                filename = os.path.join(plot_dir, self.base_filename + '.png')
                plot.figure.savefig(filename, bbox_inches='tight');
        
    def send_email(self, model_info, enable=True):
        if (enable):
            config = ConfigParser()
            config.read(os.path.join(self.project_dir, 'config/mail_config.ini'))
            server = config.get('mailer','server')
            port = config.get('mailer','port')
            login = config.get('mailer','login')
            password = config.get('mailer', 'password')
            to = config.get('mailer', 'receiver')

            subject = 'Experiment execution [' + self.model_descr + ']'
            text = 'This is an email message to inform you that the python script has completed.'
            message = text + '\n' + str(self.get_time_info()) + '\n' + str(model_info)

            smtp = smtplib.SMTP_SSL(server, port)
            smtp.login(login, password)

            body = '\r\n'.join(['To: %s' % to,
                                'From: %s' % login,
                                'Subject: %s' % subject,
                                '', message])
            try:
                smtp.sendmail(login, [to], body)
                print ('email sent')
            except Exception:
                print ('error sending the email')

            smtp.quit()
    
    def save_loss(self, train_losses, val_losses, enable=True):
        if (enable):
            losses_dir = self.__create_dir('losses')
            train_dir, val_dir = self.__create_train_val_dir_in(losses_dir)
            train_filename = os.path.join(train_dir, self.base_filename + '.txt')
            val_filename = os.path.join(val_dir, self.base_filename + '.txt')
            np.savetxt(train_filename, train_losses, delimiter=",", fmt='%g')
            np.savetxt(val_filename, val_losses, delimiter=",", fmt='%g')
            
    def save_examples(self, inputs, target, output, step):
        input_seq_length = inputs.shape[2]
        row = step // input_seq_length
        fig_input, ax_input = plt.subplots(nrows=1, ncols=input_seq_length)
        fig_ground_truth, ax_ground_truth = plt.subplots(nrows=row, ncols=input_seq_length)
        fig_prediction, ax_prediction = plt.subplots(nrows=row, ncols=input_seq_length)
        count = 0
        for i in range(row):
            for j in range(input_seq_length):
                if step == 5:
                    ax_input[j] = self.__create_image_plot(inputs, ax_input, i, j, count+j, step)  
                    ax_ground_truth[j] = self.__create_image_plot(target, ax_ground_truth, i,j ,count+j, step)
                    ax_prediction[j] = self.__create_image_plot(output, ax_prediction, i,j ,count+j, step)
                else:
                    if i == 0:
                        ax_input[j] = self.__create_image_plot(inputs, ax_input, i, j, count+j, step, ax_input=True)
                    ax_ground_truth[i][j] = self.__create_image_plot(target, ax_ground_truth, i,j ,count+j, step)
                    ax_prediction[i][j] = self.__create_image_plot(output, ax_prediction, i,j ,count+j, step)
            count+=5
                    
        examples_dir = self.__create_dir('examples')
        self.__save_image_plot(fig_input, examples_dir, 'input', step, fig_input=True)
        self.__save_image_plot(fig_ground_truth, examples_dir, 'ground_truth', step)
        self.__save_image_plot(fig_prediction, examples_dir, 'prediction', step)
    
    def get_checkpoint_filename(self):
        check_dir = self.__create_dir('checkpoints')
        filename = os.path.join(check_dir, self.base_filename + '.pth.tar')
        return filename
        
    def to_readable_time(self, timestamp):
        hours = int(timestamp / (60 * 60))
        minutes = int((timestamp % (60 * 60)) / 60)
        seconds = timestamp % 60.
        return f'{hours}:{minutes:>02}:{seconds:>05.2f}'
        
    def get_time_info(self):
        end_time = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        end_time_timestamp = tm.time()
        elapsed_time = end_time_timestamp - self.start_time_timestamp
        elapsed_time = self.to_readable_time(elapsed_time)
        time_info = {'model': self.model_descr,
                      'version': self.version,
                      'start_time': self.start_time,
                      'end_time': end_time,
                      'elapsed_time': elapsed_time}
        return time_info
       
    def get_mask_land(self):
        """
        Original chirps dataset has no ocean data, 
        so this mask is required to ensure that only land data is considered
        """
        filename = os.path.join(self.project_dir, 'data', 'chirps_mask_land.npy')
        mask_land = np.load(filename)
        mask_land = torch.from_numpy(mask_land).float()
        return mask_land
        
    @staticmethod         
    def generate_list_from(integer, size=3):
        if isinstance(integer,int):
            return [integer] * size
        return integer        
        
    def __create_train_val_dir_in(self, dir_path):
        train_dir = os.path.join(dir_path, 'train')
        os.makedirs(train_dir, exist_ok=True)
        val_dir = os.path.join(dir_path, 'val')   
        os.makedirs(val_dir, exist_ok=True)     
        return train_dir, val_dir
    
    def __create_dir(self, dir_name):
        new_dir = os.path.join(self.output_dir, dir_name, self.model_descr)
        os.makedirs(new_dir, exist_ok=True)
        return new_dir
        
    def __create_image_plot(self, tensor, ax, i, j, index, step, ax_input=False):
        cmap = 'YlGnBu' if self.base_filename.startswith('chirps') else 'viridis'
        tensor_numpy = tensor[0,:,index,:,:].squeeze().cpu().numpy()
        if step == 5 or ax_input:
            ax[j].imshow(np.flipud(tensor_numpy), cmap=cmap)
            ax[j].get_xaxis().set_visible(False)
            ax[j].get_yaxis().set_visible(False)
        else:
            ax[i][j].imshow(np.flipud(tensor_numpy), cmap=cmap)
            ax[i][j].get_xaxis().set_visible(False)
            ax[i][j].get_yaxis().set_visible(False)        
        return ax
        
    def __save_image_plot(self, figure, folder, name, step, fig_input=False):
        y = 0.7 if (step == 5 or fig_input) else 0.9
        figure.suptitle(name, y=y)
        filename = os.path.join(folder, name + '_' + self.base_filename + '.png') 
        figure.savefig(filename, dpi=300)