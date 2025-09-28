import numpy as np



class GetData:
    def __init__(self, directory, texp, config, dim_latent_main, img_size, norm_keys, input_keys, inputadd_keys, c_input, bins, y_min, wbin, phase):
        self.texp = texp
        self.config = config
        self.dim_latent_main = dim_latent_main
        self.img_size = img_size
        self.c_input = c_input
        self.bins = bins
        self.y_min = y_min
        self.wbin = wbin
        self.phase = phase
        
        ##### User-defined catalog and multi-band cutout images #####
        # The catalog ("catalog.npz") contains digit IDs for the training, test and validation samples,
        # and catalog data including stellar mass (lgm_tot_p50), magnitudes, magnitude errors and galactic reddening E(B-V), etc.
        
        # The images ("images.npz") in different bands are restored separately, and should correspond to the catalog row by row, indexed by the IDs.
        # All the images in a band are loaded at once. If not, the "load_img" method (see below) should be redefined and used instead.
        
        # Dimensions of each property/variable in the catalog: [n_sample]
        # Dimensions of images in each band: [n_sample, img_size, img_size]
        # Number of bands == c_input
        
        catalog = np.load(directory + 'catalog.npz')

        id_train = catalog['id_train']
        id_validation = catalog['id_validation']
        id_test = catalog['id_test']
        self.yt = catalog['lgm_tot_p50']  # stellar mass
        
        if self.config == 0:  # photometry-only
            self.input = np.stack([catalog[k] for k in input_keys], 1)  # [n_sample, c_input]
        
        elif self.config == 1:  # image-based
            images = np.load(directory + 'images.npz')
            
            if norm_keys != None:
                min_ercentile = [2, 0.1, 0.1, 0.1, 0.1]  # five optical bands
                images_norm = []
                for i in range(5):
                    minflux, maxflux = np.percentile(catalog[norm_keys[i]], [min_ercentile[i], 100])
                    images_norm.append(images[input_keys[i]] / np.expand_dims(np.expand_dims(np.clip(catalog[norm_keys[i]], minflux, maxflux), 1), 1))
                self.input = np.stack(images_norm, -1)  # [n_sample, img_size, img_size, c_input]
            else:
                self.input = np.stack([images[k] for k in input_keys], -1)  # [n_sample, img_size, img_size, c_input]
           
        self.inputadd = np.stack([catalog[k] for k in inputadd_keys], 1)  # [n_sample, c_inputadd]
                
        self.ylist = (0.5 + np.arange(self.bins)) * self.wbin + self.y_min
        
        self.id_train = id_train
        self.id_validation = id_validation
        self.id_test = id_test
        self.n_train = len(id_train)
        self.n_validation = len(id_validation)
        self.n_test = len(id_test)
        print ('Training,Validation,Test:', self.n_train, self.n_validation, self.n_test)



    def load_img(self, id_):
        ##### User-defined image load method (for each galaxy) #####
        # Should be used if all the images in a dataset cannot be loaded at once
        img = np.copy(self.input[id_])
        return img
          


    def img_rescale(self, img):
        index_neg = img < 0
        index_pos = img > 0
        img[index_pos] = np.log(img[index_pos] + 1.0)
        img[index_neg] = -np.log(-img[index_neg] + 1.0)
        return img



    def img_reshape(self, img):
        mode = np.random.random()
        if mode < 0.25: img = np.rot90(img, 1)
        elif mode < 0.50: img = np.rot90(img, 2)
        elif mode < 0.75: img = np.rot90(img, 3)
        else: pass
            
        mode = np.random.random()
        if mode < 0.5: img = np.flip(img, 0)
        else: pass            
        return img
    


    def img_morph_aug(self, img):
        mode = np.random.random()
        if mode < 0.25: img = np.rot90(img, 1)
        elif mode < 0.50: img = np.rot90(img, 2)
        elif mode < 0.75: img = np.rot90(img, 3)
            
        if mode > 0.75: mode = 0
        else: mode = np.random.random()
        if mode < 0.5: img = np.flip(img, 0)
        return img



    def get_ystats(self, yest_q, yt_q, yprob_q, y_q):
        delta = yest_q - yt_q
        residual = np.mean(delta)
        sigma_mad = 1.4826 * np.median(abs(delta - np.median(delta))) 
        eta = len(delta[abs(delta) - 3 * sigma_mad > 0]) / float(len(delta))
        crps = np.mean(np.sum((np.cumsum(yprob_q, 1) - np.cumsum(y_q, 1)) ** 2, 1)) * self.wbin                        
        return residual, sigma_mad, eta, crps



    def get_ypoints(self, yprob_q, n_sample):
        yest_mean = np.sum(yprob_q * np.expand_dims(self.ylist, 0), 1)

        yest_mode = np.zeros(n_sample)
        for i in range(n_sample):
            yest_mode[i] = self.ylist[np.argmax(yprob_q[i])]

        yest_median = np.zeros(n_sample)
        for i in range(n_sample):
            yest_median[i] = self.ylist[np.argmin(abs(np.cumsum(yprob_q[i]) - 0.5))]
        return yest_mean, yest_mode, yest_median
                


    def get_batch_data(self, id_):
        if self.phase == 0:  # training
            id_batch = self.id_validation#[:2000]
        elif self.phase == 1:  # inference
            id_batch = id_
           
        if self.config == 0:  # photometry-only
            input_batch = self.input[id_batch]       
        elif self.config == 1:  # image-based
            input_batch = np.zeros((len(id_batch), self.img_size, self.img_size, self.c_input))  # rescaled images as inputs
            for i in range(len(id_batch)):
                img = self.load_img(id_batch[i])
                input_batch[i] = self.img_rescale(img)
         
        inputadd_batch = self.inputadd[id_batch]
        yt_batch = self.yt[id_batch]
        
        y_batch = np.zeros((len(id_batch), self.bins))  # one-hot labels
        for i in range(len(id_batch)):
            z_index = max(0, min(self.bins - 1, int((yt_batch[i] - self.y_min) / self.wbin)))
            y_batch[i, z_index] = 1.0
        return input_batch, yt_batch, y_batch, inputadd_batch
    
    
    
    def get_cross_entropy_indiv(self, yprob_q, y_q):
        return -1 * np.sum(y_q * np.log(yprob_q + 10**(-20)), 1)
    
    
    
    def get_cost_y_stats(self, data_q, session, x, y, inputadd, x2, y2, inputadd2, p_set, ce_set, latent_set):
        if self.phase == 0:  # training
            input_q, yt_q, y_q, inputadd_q = data_q
            n_sample = len(yt_q)
        elif self.phase == 1:  # inference
            id_all = np.concatenate([self.id_train, self.id_validation, self.id_test])
            n_sample = len(id_all)
            cross_entropy_indiv_q = np.zeros(n_sample)
            latent_q = np.zeros((n_sample, self.dim_latent_main))
                
        yprob_q = np.zeros((n_sample, self.bins))
        cross_entropy_avg = 0
        batch = 512
        
        for i in range(0, n_sample, batch):
            index_i = np.arange(i, min(i + batch, n_sample))
            if self.phase == 0:
                input_batch = input_q[index_i]                
                y_batch = y_q[index_i]
                inputadd_batch = inputadd_q[index_i]
            elif self.phase == 1:
                input_batch, _, y_batch, inputadd_batch = self.get_batch_data(id_all[index_i])
                            
            feed_dict = {x:input_batch, y:y_batch, inputadd:inputadd_batch}
            output_batch = session.run(p_set + ce_set + latent_set, feed_dict = feed_dict)
            
            yprob_q[index_i] = output_batch[0]
            cross_entropy_avg = cross_entropy_avg + output_batch[1] * len(index_i)                
            if self.phase == 1:
                latent_q[index_i] = output_batch[2]
                cross_entropy_indiv_q[index_i] = self.get_cross_entropy_indiv(yprob_q[index_i], y_batch)
        cross_entropy_avg = cross_entropy_avg / n_sample
                       
        if self.phase == 0:
            yest_mean = np.sum(yprob_q * np.expand_dims(self.ylist, 0), 1)
            residual, sigma_mad, eta, crps = self.get_ystats(yest_mean, yt_q, yprob_q, y_q)   
            return cross_entropy_avg, residual, sigma_mad, eta, crps
        elif self.phase == 1:
            yest_mean, yest_mode, yest_median = self.get_ypoints(yprob_q, n_sample)
            return cross_entropy_avg, cross_entropy_indiv_q, latent_q, yest_mean, yest_mode, yest_median


    
    def get_id_nonoverlap(self, id_all, id_pre, subbatch):
        id_select = np.random.choice(id_all, subbatch)
        for i in range(subbatch):
            while id_select[i] == id_pre[i]:
                id_select[i] = np.random.choice(id_all)
        return id_select
    
    
        
    def get_next_subbatch(self, subbatch):
        if self.texp == 0:  # mutual information estimation
            id1_subbatch = np.random.choice(self.id_train, subbatch)
            id_list = [id1_subbatch]
            get_aug = [False]
        elif self.texp == 1:  # supervised contrastive learning for causal analysis
            id1_subbatch = np.random.choice(self.id_train, subbatch)
            id2_subbatch = self.get_id_nonoverlap(self.id_train, id1_subbatch, subbatch) 
            id_list = [id1_subbatch, id2_subbatch]
            get_aug = [True, True]
            
        input_list = []
        y_list = []
        inputadd_list = []
        
        for k, id_subbatch in enumerate(id_list):        
            id_subbatch = np.array(id_subbatch)
            yt_subbatch = self.yt[id_subbatch]
            inputadd_subbatch = self.inputadd[id_subbatch]
            
            y_subbatch = np.zeros((subbatch, self.bins))  # one-hot labels
            for i in range(subbatch):
                yt_index = max(0, min(self.bins - 1, int((yt_subbatch[i] - self.y_min) / self.wbin)))
                y_subbatch[i, yt_index] = 1.0
                
            if self.config == 0:  # photometry-only
                input_subbatch = self.input[id_subbatch] 
                input_subbatch = [input_subbatch]
            elif self.config == 1:  # image-based
                input_subbatch = np.zeros((subbatch, self.img_size, self.img_size, self.c_input))
                input_subbatch_morph = np.zeros((subbatch, self.img_size, self.img_size, self.c_input))
                for i in range(subbatch):
                    img = self.load_img(id_subbatch[i])               
                    img = self.img_reshape(img)
                    input_subbatch[i] = self.img_rescale(np.copy(img))
                    
                    if get_aug[k]:
                        img_morph = self.img_morph_aug(np.copy(img))
                        input_subbatch_morph[i] = self.img_rescale(img_morph)

                input_subbatch = [input_subbatch, input_subbatch_morph]
            input_list.append(input_subbatch)
            y_list.append(y_subbatch)
            inputadd_list.append(inputadd_subbatch)
        return input_list, y_list, inputadd_list

      