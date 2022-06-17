# Probabilitybuckets
# reimplemented by David Sommer (david_sommer@inf.ethz.ch)
# according to
#      "Privacy Buckets: Upper and Lower Bounds for r-Fold Approximate Differential Privacy"
#      (https://eprint.iacr.org/2017/1034, version 2018 May 8th)

import logging
import numpy as np
import copy
import gc
import os
import pickle
import errno
import xxhash

_infty_bucket_warning_bound = 1e-5
_virtual_error_warning_bound = 1e-3

class ProbabilityBuckets:

    def __init__(self,
                 number_of_buckets = 100000,
                 factor = None,
                 dist1_array = None,
                 dist2_array = None,
                 caching_directory = None,
                 free_infty_budget = 10**(-20),
                 logging_level = logging.INFO,
                 error_correction = True,
                 skip_bucketing = False,
                 **kwargs):

        self.logging_level = logging_level
        self.logger_setup(level=self.logging_level)

        for key in kwargs:
            self.logger.warning( "Warning: option {} not implemented".format(key) )

        # infty_bucket is excluded
        self.number_of_buckets = self.number_of_buckets = int(4 * (number_of_buckets // 4) ) + 1
        self.factor = np.float64(factor)
        self.log_factor = np.log(factor, dtype=np.float64)

        # arbitrarily chosen
        self.squaring_threshold_factor = np.float64(1.1)
        self.free_infty_budget = np.float64(free_infty_budget)

        # in case of skip_bucketing = True, caching_setup() has to be called by the derived class as caching depends on a filled self.bucket_distribution
        self.caching_super_directory = caching_directory
        self.caching_directory = None   # will be set inside self.caching_setup() if self.caching_super_directory != None

        # skip bucketing if something else (e.g. derived class) creates buckets
        if not skip_bucketing:
            self.create_bucket_distribution(dist1_array, dist2_array, error_correction)
            self.caching_setup()

    def caching_setup(self):
        # setting up caching. Hashing beginning bucket_distribution to avoid name collisions
        if self.caching_super_directory:
            hasher = xxhash.xxh64(self.bucket_distribution, seed=0)
            hasher.update(str(self.error_correction))
            hasher.update(str(self.free_infty_budget))
            array_name = hasher.hexdigest()

            self.caching_directory = os.path.join(self.caching_super_directory, array_name)
            self.logger.info("Caching directory: {}".format(self.caching_directory))

    def logger_setup(self, level):
        self.logger = logging.getLogger(__name__)  # all instances use the same logger. Randomize the name if not appreciated
        if not len(self.logger.handlers):
            self.logger.setLevel(level=level)
            ch = logging.StreamHandler()
            ch.setLevel(level=level)
            ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s: %(message)s'))
            self.logger.addHandler(ch)

    def create_bucket_distribution(self, distr1, distr2, error_correction):
        #
        # For explanation of variables, see comment at the beginning of method "self.compose_with"
        #

        self.logger.info("Create bucket distribution")
        assert len(distr1) == len(distr2)

        distr1 = np.array(distr1, dtype=np.float64)
        distr2 = np.array(distr2, dtype=np.float64)

        self.bucket_distribution = np.zeros(self.number_of_buckets, dtype=np.float64)
        infty_mask = (distr2 == 0)
        null_mask = (distr1 == 0)

        indices = np.ma.log(np.divide(distr1, distr2, where=~infty_mask))/self.log_factor + self.number_of_buckets//2
        indices = np.ceil(indices).astype(int)

        # set up error correction
        self.error_correction = error_correction
        virtual_error = np.zeros(self.number_of_buckets, dtype=np.float64)

        if self.error_correction:
            # we want errors(x,i) = P_B(x) - P_A(x) / f**i  =  P_B(x) - exp( log( P_A(x) ) - i * log_factor )
            errors = distr2 - distr1 / self.factor**(indices - self.number_of_buckets//2)
            # errors = distr2 - np.exp(np.ma.log(distr1, dtype=np.float64) - (indices - self.number_of_buckets//2)   * self.log_factor, dtype=np.float64)
        else:
            errors = np.zeros(len(distr1), dtype=np.float64)

        # fill buckets
        self.infty_bucket = np.float64(0.0)
        self.distinguishing_events = np.float64(0.0)
        for i, m_infty, m_null, a, err in zip(indices, infty_mask, null_mask, distr1, errors):
            if m_infty:
                self.distinguishing_events += a
                # self.infty_bucket += a
                continue
            if m_null:
                continue
            # i = int(np.ceil(i))
            if i >= self.number_of_buckets:
                self.infty_bucket += a
                continue
            if i < 0:
                self.bucket_distribution[0] += a
                virtual_error[0] += err
                continue
            self.bucket_distribution[i] += a
            virtual_error[i] += err

        self.one_index = int(self.number_of_buckets // 2)
        self.u = np.int64(1)

        if self.error_correction:
            self.virtual_error = virtual_error
            self.real_error = self.virtual_error.copy()
            self.real_error[0] = 0.0

        if self.infty_bucket > _infty_bucket_warning_bound:
            self.logger.warning("Infty bucket (numerical errors) is above {:g}. "
                "This error will exponentiate over compositions. "
                "Decrease factor or increase number of buckets to avoid this.".format(_infty_bucket_warning_bound))

    def squaring(self):
        self.logger.info("Squaring")

        # collapse puts every two consecutive buckets in a single bucket
        collapse = lambda arr: np.sum(arr.reshape( len(arr) // 2, 2), axis=1)

        if self.error_correction:
            self.logger.debug("  Error correction.")
            # n_half_p1_to_n_half = [ -n/2+1, .. , n/2 ]
            # By adding self.one_index to this array, we get the corresponding self.bucket_distribution indices.
            # assumptions on next line: self.one_index % 2 == 1 and (self.number_of_buckets - 1) % 4 == 0
            n_half_p1_to_n_half = np.arange(- (self.one_index//2) + 1, self.one_index // 2 + 1)
            n_half_p1_to_n_half_addr = slice(n_half_p1_to_n_half[0] + self.one_index, n_half_p1_to_n_half[-1] + self.one_index + 1)
            assert -n_half_p1_to_n_half[0] + 1 == n_half_p1_to_n_half[-1]   # sanity check. If failing: num_of_buckets invalid

            # div(i) = (1/f**(2i-1) - 1/f**(2i) )
            div = lambda i: (1.0 / self.factor**( 2 * i - 1) ) - ( 1.0 / self.factor**( 2 * i ) )
            div_arr = div( n_half_p1_to_n_half )

            def square_error(array):
                temp_error = np.zeros(self.number_of_buckets, dtype=np.float64)

                temp_error[n_half_p1_to_n_half_addr] = collapse(array[1:])

                # here we add B(i) * (1/f**(2i-1) - 1/f**(2i) ) for every i in [-n/2+1, n/2]
                temp_error[n_half_p1_to_n_half_addr] += self.bucket_distribution[ (n_half_p1_to_n_half * 2) - 1 + self.one_index] * div_arr
                return temp_error

            self.real_error = square_error(self.real_error)

            temp_virtual_error = square_error(self.virtual_error)
            temp_virtual_error[self.one_index // 2] = self.virtual_error[0]
            self.virtual_error = temp_virtual_error

            self.u += 1

        temp_bucket_distribution = np.zeros(self.number_of_buckets)
        lower_half, upper_half = self.split_array_equally(self.bucket_distribution)

        temp_bucket_distribution[self.one_index // 2:self.one_index] = collapse(lower_half)
        temp_bucket_distribution[self.one_index + 1:self.one_index + self.one_index // 2 + 1] = collapse(upper_half)
        temp_bucket_distribution[self.one_index] = self.bucket_distribution[self.one_index]

        self.bucket_distribution = temp_bucket_distribution
        self.log_factor *= 2
        self.factor = np.exp(self.log_factor)

        gc.collect()

    def opt_compose_with(self, probability_buckets, after_squaring = False, threshold_factor = None):
        assert(after_squaring == False and threshold_factor == None), "Function is being called in an unsupported way. We should fix this."
        return self.compose_with(probability_buckets)

    def compose_with(self, pb, allow_modify_instance=False):
        self.logger.info("Composing")
        assert self.number_of_buckets == pb.number_of_buckets
        assert self.factor == pb.factor, "ERROR, trying to compose distributions with different factors"

        # we want to compute the product of the two probability buckets instances A and B: I_A.I_B  ( . denotes a "inner" product)
        # let A (or B) denote the bucketlist and A_infty (or B_infty) denote the infinty bucket: I_A = (A, A_infty), I_B = (B, B_infty)
        # so, sligthly informal, we do
        #
        # I_A.I_B = (A, A_infty).(B, B_infty)
        #         = A.B + sum(A) * B_infty + A_infty * sum(B) + A_infty * B_infty             ( * denotes scalar multiplication)
        #         = A.B + (1-A_infty) * B_infty + A_infty * (1 - B_infty) + A_infty * B_infty ( using 1 = sum(B) + B_infty)
        #         = A.B + A_infty + B_infty - A_infty * B_infty
        #
        # as in the computation of A.B under- and overflows can occur, we get in the end six terms:
        #
        # I_A.I_B = (A.B)_under + A.B + (A.B)_over + A_infty + B_infty - A_infty * B_infty
        #
        # The first term goes to the first bucket, the second is the bucket distribution, and the last 4 terms we pack in the infty bucket
        #
        # UPDATE: there are now two "infty buckets":
        #                   self.distinguishing_events - keeps track of the real distinguishing events
        #                   self.infty_bucket - containing numerical overswap of A.B, and the mix terms of disting_events and infty_bucket.
        #         Technically, there are two terms added to self.infty_bucket: -A_infty*B_dist - A_dist*B_infty
        #         resulting from (A, A_infty, A_dist).(B, B_infty, B_dist) .

        # add "A_dist + B_dist - A_dist * B_dist"
        self.distinguishing_events = self.distinguishing_events + pb.distinguishing_events - (self.distinguishing_events * pb.distinguishing_events)
        # "A_infty + B_infty - A_infty * B_infty - A_infty*B_dist - A_dist*B_infty"
        temp_infty_bucket = np.float64(0)
        temp_infty_bucket += self.infty_bucket + pb.infty_bucket - (self.infty_bucket * pb.infty_bucket)
        temp_infty_bucket += - self.infty_bucket * pb.distinguishing_events - self.distinguishing_events * pb.infty_bucket

        temp_bucket_distribution = np.zeros(self.number_of_buckets, dtype=np.float64)

        while True:
            delta_infty_bucket = np.float64(0.0)

            # calculate (A.B)_over
            self.logger.info("   Compute (A.B)_over")
            temp_over = self.convolve_full(self.bucket_distribution[self.one_index+1:], pb.bucket_distribution[self.one_index+1:])

            # add all infty parts together: "(A.B)_over + A_infty + B_infty - A_infty * B_infty"
            delta_infty_bucket = np.sum(temp_over[self.one_index-1:]) + temp_infty_bucket

            max_growth_allowed = self.squaring_threshold_factor * (self.infty_bucket + pb.infty_bucket)
            if delta_infty_bucket >= self.free_infty_budget and delta_infty_bucket > max_growth_allowed:
                self.squaring()
                if self.bucket_distribution is not pb.bucket_distribution:
                    if not allow_modify_instance:  # make a copy so we do not change the original instance
                        pb = pb.copy()
                        allow_modify_instance = True
                    pb.squaring()
                continue
            break

        # compute all intermediate buckets "A.B"
        self.logger.info("   Compute (A.B)")
        temp_bucket_distribution[1:] = self.convolve_same(self.bucket_distribution, pb.bucket_distribution)[1:]

        # compute the first bucket (A.B)_under
        self.logger.info("   Compute (A.B)_under")
        temp_under = self.convolve_full(self.bucket_distribution[0:self.one_index+1], pb.bucket_distribution[0:pb.one_index+1])
        temp_under = np.sum(temp_under[0:self.one_index+1])
        temp_bucket_distribution[0] = max(0,temp_under)

        if self.error_correction:
            assert pb.error_correction
            convert_to_B = lambda distribution, factors: distribution / factors
            # factors = self.factor ** np.arange(-self.one_index, self.one_index + 1 )  # how numerically stable is that?
            factors = np.exp(self.log_factor * np.arange(-self.one_index, self.one_index + 1 ) )

            temp_buck_distr_B_self = convert_to_B(self.bucket_distribution, factors)
            temp_buck_distr_B_pb = convert_to_B(pb.bucket_distribution, factors)
            temp_buck_distr_B_convolved = convert_to_B(temp_bucket_distribution, factors)

            # As l(i) = sum_{k+j=i} B^A_j/f**j * l^B(k) + B^A_k/f**k * l^B(j) + l^A(j) * l^B(k)
            #         = sum_{k+j=i} ( B^A_j/f**j + l^A(j) ) * ( B^B_k/f**k + l^B(k) ) - B^A_j/f**j * B^B_k/f**k
            # We compute the latter one because it involves only one convolution and the substraction term we already know

            self.logger.info("   Compute real_error A.B")
            self.real_error = self.convolve_same(temp_buck_distr_B_self + self.real_error, temp_buck_distr_B_pb + pb.real_error) - temp_buck_distr_B_convolved
            self.real_error[0] = 0

            self.logger.info("   Compute virtual_error A.B")
            temp_buck_distr_B_virt_err_self = temp_buck_distr_B_self + self.virtual_error
            temp_buck_distr_B_virt_err_pb = temp_buck_distr_B_pb + pb.virtual_error
            temp_virtual_error = self.convolve_same(temp_buck_distr_B_virt_err_self, temp_buck_distr_B_virt_err_pb) - temp_buck_distr_B_convolved

            self.logger.info("   Compute virtual_error A.B_under")
            temp_virtual_error_under = self.convolve_full(temp_buck_distr_B_virt_err_self[0:self.one_index+1], temp_buck_distr_B_virt_err_pb[0:pb.one_index+1])
            temp_virtual_error_under = np.sum(temp_virtual_error_under[0:self.one_index+1])
            temp_virtual_error[0] = max(0, temp_virtual_error_under)

            self.virtual_error = temp_virtual_error

            self.u = self.u + pb.u

            # decrease reference counters
            del temp_buck_distr_B_self
            del temp_buck_distr_B_pb
            del temp_buck_distr_B_convolved
            del temp_buck_distr_B_virt_err_self
            del temp_buck_distr_B_virt_err_pb
            del temp_virtual_error_under

        # overwrites
        self.bucket_distribution = temp_bucket_distribution
        self.infty_bucket = delta_infty_bucket

        # clean up
        del temp_over
        del temp_under
        gc.collect()

        return pb

    @classmethod
    def load_state(cls, state_directory):

        params = pickle.load(open(os.path.join(state_directory, "non_ndarray"),'rb'))

        # If you search bugs with incomplete loaded states, look here.
        # The "instance.__init__()"  method is *NOT* called when loading!
        instance = cls.__new__(cls)
        instance.__dict__.update(params)
        instance.logger_setup(instance.logging_level)

        instance.bucket_distribution = np.fromfile(os.path.join(state_directory, "bucket_distribution"), dtype=np.float64)
        if instance.error_correction:
            instance.real_error = np.fromfile(os.path.join(state_directory, "real_error"), dtype=np.float64)
            instance.virtual_error = np.fromfile(os.path.join(state_directory, "virtual_error"), dtype=np.float64)

        return instance

    def save_state(self, state_directory):
        self.mkdir_p(state_directory)

        excluded_attributes = ['logger', 'bucket_distribution', 'real_error', 'virtual_error']

        # excluding np arrays from dumping in "non_ndarray" without copying large arrays:
        save_dict = {}
        for key in self.__dict__.keys():
            if key not in excluded_attributes:
                save_dict[key] = self.__dict__[key]

        pickle.dump(save_dict, open(os.path.join(state_directory, "non_ndarray"),'wb'))

        self.bucket_distribution.tofile(open(os.path.join(state_directory, "bucket_distribution"), 'w'))
        if self.error_correction:
            self.real_error.tofile(open(os.path.join(state_directory, "real_error"), 'w'))
            self.virtual_error.tofile(open(os.path.join(state_directory, "virtual_error"), 'w'))

    def compose(self, nr_of_compositions):
        if not self.caching_directory:
            self.logger.critical("This method requires caching. Abort")
            return None

        state_filepath_base = os.path.join(self.caching_directory, "compositions-")
        get_state_filepath = lambda needed_exp: state_filepath_base + str(int(2**needed_exp))
        target_state_filepath = state_filepath_base + str(nr_of_compositions)

        if os.path.isdir(target_state_filepath):
            self.logger.info("Target state is cached. Loading it")
            return self.load_state(target_state_filepath)

        # wich compositions do we need
        target_exp = int( np.log2(nr_of_compositions) )
        needed_compositions = [ x if (nr_of_compositions & (2**x) != 0) else -1 for x in range(target_exp + 1)]
        needed_compositions = list(filter(lambda a: a != -1, needed_compositions))

        self.logger.info("Needed compositions: " + ", ".join(map(str, needed_compositions)))

        # start with a copy of the current state
        previous_state = self.copy()
        avoided_self_composition = False

        # which compositions already exist? Generate?
        for needed_exp in range(target_exp + 1):
            state_filepath = get_state_filepath(needed_exp)
            if not os.path.isdir(state_filepath):
                self.logger.info("[*] State 2**" + str(needed_exp) + " does not exist. Creating it.")
                if not needed_exp == 0:
                    if avoided_self_composition:  # only load from disk when it differs from current state
                        previous_state_filepath = get_state_filepath(needed_exp - 1)
                        previous_state = self.load_state(previous_state_filepath)
                        avoided_self_composition = False
                    previous_state.compose_with(previous_state)
                    previous_state.print_state()
                    # previous_state.print_state()
                previous_state.save_state(state_filepath)
                gc.collect()
            else:
                avoided_self_composition = True
        self.logger.info("[*] All intermediate states up to 2**" + str(target_exp) + " exist now")

        previous_state = self.load_state( get_state_filepath(needed_compositions[0]) )
        self.logger.info("[*] Loaded state 2**" + str(needed_compositions[0]))

        # compose to the desired state
        for i in needed_compositions[1:]:
            self.logger.info("[*] Compose with state 2**{}".format(i))
            current_state = self.load_state(get_state_filepath(i))
            # while the factor of previous state and current state is not same
            while(current_state.factor != previous_state.factor):
                self.logger.info("factors are unequal ( {} != {} ), squaring".format(current_state.factor, previous_state.factor))
                if current_state.factor > previous_state.factor:
                    previous_state.squaring()
                else:
                    current_state.squaring()
            # now the factor should be the same
            previous_state.compose_with(current_state)

        previous_state.print_state()
        previous_state.save_state(target_state_filepath)  # caching..
        return previous_state

    def print_state(self):
        """ prints some information about the current state """
        sum_of_bucket_distr = np.sum(self.bucket_distribution)
        summary = (     'Summary:\n'
                        '   caching_directoy   : {}\n'
                        '   number_of_buckets  : {}\n'
                        '   factor             : {}\n'
                        '   infty_bucket       : {}  (max 1.0, numerical error, should be < {:g})\n'
                        '   disting_events     : {}  (max 1.0)\n'
                        '   minus-n-bucket     : {}\n'
                        '   sum bucket_distr   : {:.30f}\n'
                        '   sum of all buckets : {:.30f}  (should be 1.000000)\n'
                        '   delta_upper(eps=0) : {}'
                        .format(
                            self.caching_directory,
                            self.number_of_buckets,
                            self.factor,
                            self.infty_bucket,
                            _infty_bucket_warning_bound,
                            self.distinguishing_events,
                            self.bucket_distribution[0],
                            sum_of_bucket_distr,
                            sum_of_bucket_distr + self.infty_bucket + self.distinguishing_events,
                            self.delta_of_eps_upper_bound(0),
                        ))
        if self.error_correction:
            summary += ('\n'
                        '   delta_lower(eps=0) : {}\n'
                        '   sum(virtual_error) : {}   (max 1.0, should be < {:g})\n'
                        '   sum(real_error)    : {}   (max 1.0, should be < {:g})\n'
                        '   the u              : {}'
                        .format(
                            self.delta_of_eps_lower_bound(0),
                            np.sum(self.virtual_error),
                            _virtual_error_warning_bound,
                            np.sum(self.real_error),
                            _virtual_error_warning_bound,
                            self.u,
                        ))

        self.logger.info(summary)

    def convolve_same(self, x, y):
        return np.convolve(x, y, mode = 'same')

    def convolve_full(self, x, y):
        return np.convolve(x, y, mode = 'full')

    def delta_PDP(self, eps):
        """ Returns upper bound for tight delta for probabilistic differential privacy. Error correction not supported. """
        if self.error_correction:
            self.logger.warning("Error correction for PDP delta not supported. Omit error correction.")

        k = int(np.floor(eps / self.log_factor))

        if k > self.number_of_buckets // 2:  # eps is above of bucket_distribution range
            return self.infty_bucket + self.distinguishing_events
        if k < -self.number_of_buckets // 2:  # eps is below of bucket_distribution range
            k = -self.number_of_buckets // 2

        return np.sum(self.bucket_distribution[self.one_index + k + 1:]) + self.infty_bucket + self.distinguishing_events

    def _g_func(self, l):
        return (1 - self.factor**-l)

    def delta_ADP(self, eps):
        """ Returns an upper bound for tight delta for approximate differntial privacy. Error correction is supported."""
        return self.delta_ADP_upper_bound(eps)

    def delta_ADP_upper_bound(self, eps):
        # Use np.floor to guarantee an upper bound for the delta(eps) graph
        k = int(np.floor(eps / self.log_factor))

        if k > self.number_of_buckets // 2:  # eps is above of bucket_distribution range
            return self.infty_bucket + self.distinguishing_events
        if k < -self.number_of_buckets // 2:  # eps is below of bucket_distribution range
            k = -self.number_of_buckets // 2

        ret = np.sum(self._g_func(np.arange(1, self.one_index - k + 1)) * self.bucket_distribution[self.one_index + k + 1:])
        if self.error_correction:
            ret -= np.exp(eps) * np.sum( self.real_error[ min(self.one_index + k + self.u, self.number_of_buckets): ] )

        # check:
        # a = np.sum(self._g_func(np.arange( self.one_index - k)) * self.bucket_distribution[self.one_index + k + 1:][self.u-1:])
        # a -= np.exp(eps) * np.sum( self.real_error[ min(self.one_index + k + self.u, self.number_of_buckets): ] )
        # assert np.all( a >= 0)
        return ret + self.infty_bucket + self.distinguishing_events

    def delta_ADP_lower_bound(self, eps):
        if not self.error_correction:
            self.logger.error("Error correction required for lower bound")
            return None
        # Use np.ceil to garantee an lower bound for the delta(eps) graph
        k = int(np.ceil(eps / self.log_factor))

        if k > self.number_of_buckets // 2:  # eps is above of bucket_distribution range
            return self.distinguishing_events
        if k <= -self.number_of_buckets // 2:  # eps is below of bucket_distribution range
            k = -self.number_of_buckets // 2 + 1

        vals = self._g_func(np.arange(1, self.one_index - k + 1)) * self.bucket_distribution[self.one_index + k + 1:]
        vals -= np.exp(eps) * self.virtual_error[self.one_index + k + 1:]
        vals[vals < 0] = 0  # = max(0, vals)
        return np.sum(vals) + self.distinguishing_events

    def renyi_divergence_upper_bound(self, alpha):
        """
        returns a upper bound on the alpha renyi-divergence for a given alpha >= 1

        R(aplha) = 1/(alpha - 1) * log_e E_{x~B} (A/B)^(alpha)    if alpha > 1
        R(aplha) = E_{x~B} log_e (A/B)^(alpha)                    if alpha == 1
        """

        assert self.distinguishing_events == 0 and self.infty_bucket == 0, \
                "Nonzero infty bucket or distingushing events not supported"

        # if alpha == 1, the divergence is reduced to KL-divergence
        if alpha == 1:
            return self.KL_divergence()

        # else, compute the renyi-divergence
        lam = alpha - 1

        # the alpha renyi moments are the alpha-1 moments of the exponentiated bucket_distribution. Informal,
        # i.e. R(alpha) = 1/lam * ln E_{y~buckets} exp(y*lam)
        #
        # for additional details, see Lemma 8 in
        # Sommer et al. "Privacy loss classes: The central limit theorem in differential privacy." PoPETS 2019.2

        # to provide a upper bound, we assume that all content of a specific bucket manifests at the position
        # with the highest leakage (most right)
        summands = np.exp((np.arange(self.number_of_buckets) - self.one_index)*self.log_factor*lam)
        expectation_value = np.sum(self.bucket_distribution*summands)
        renyi_div = np.log(expectation_value) / lam

        return renyi_div

    def KL_divergence(self):
        return np.sum(self.bucket_distribution * ((np.arange(self.number_of_buckets) - self.one_index)*self.log_factor))

    # needed for pickle (and deepcopy)
    def __getstate__(self):
        d = self.__dict__.copy()  # copy the dict since we change it
        del d['logger']           # remove logger instance entry
        return d

    def __setstate__(self, dict):
        if not hasattr(self, 'logger'):
            self.logger_setup(dict['logging_level'])
        self.__dict__.update(dict)

    # utility functions

    def copy(self):
        # I hope this is sufficient
        return copy.deepcopy(self)

    def mkdir_p(self, path):
        try:
            os.makedirs(path)
        except OSError as exc:  # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(path):
                pass
            else:
                raise exc

    def split_array_equally(self, arr):
        """
        Splits array in a upper and lower part. Assumes len(arr) % 2 == 1
        """
        mid = len(arr) // 2
        return arr[:mid], arr[mid + 1:]

    #
    # DEPRECATED method names
    #

    def delta_of_eps(self, eps):
        """Deprecated. Use '.delta_ADP_upper_bound(eps)'"""
        return self.delta_ADP_upper_bound(eps)

    def delta_of_eps_upper_bound(self, eps):
        """Deprecated. Use '.delta_ADP_upper_bound(eps)'"""
        return self.delta_ADP_upper_bound(eps)

    def delta_of_eps_lower_bound(self, eps):
        """Deprecated. Use '.delta_ADP_lower_bound(eps)'"""
        return self.delta_ADP_lower_bound(eps)


class ProbabilityBuckets_fromDelta(ProbabilityBuckets):
    """
    Instructions besides the use of the parent ProbabilityBuckets class (which can be found there) can be found at
    ProbabilityBuckets_fromDelta.create_bucket_distribution
    """
    def __init__(self, delta_func, DP_type, **kwargs):
        """
        delta_func: callable that returns a tight ADP or PDP delta for (positive and NEGATIVE) epsilons
        DP_type = 'adp' | 'pdp' for aproximate or probabilistic differential privacy respectivly.
        """

        # Tell the parent __init__() method that we create our own bucket_distribution.
        kwargs['skip_bucketing'] = True

        # Our custom create_bucket_distribution() method does not set up the error correction
        if 'error_correction' in kwargs and kwargs['error_correction']:
            raise NotImplementedError("Error correction not supported.")
        kwargs['error_correction'] = False

        super(ProbabilityBuckets_fromDelta, self).__init__(**kwargs)

        self.create_bucket_distribution(delta_func, DP_type)

        # Caching setup needs to be called after the buckets have been filled as the caching utilized a hash over the bucket distribution
        self.caching_setup()

    def create_bucket_distribution(self, delta_func, DP_type):
        """
        This foo fits a bucket_distribution to a delta_func(epsilon) function using a non-negative least squares fit.
        'delta_func' needs to accept negative epsilons as well. Error correction is not supported.
        It makes use of the defintion of tight approximate differential privacy (see [1])

            delta(epsilon) = bucket(infinity) + sum_{y_i>epsislon}  (1 - exp(eps - y_i)) bucket(y_i)

        or in the probablistic differential privacy case

            delta(epsilon) = bucket(infinity) + sum_{y_i>epsislon} bucket(y_i)

        For 'number_of_buckets + 2' epsilons, we create a matrix G containing the terms (1 - exp(eps - y_i)) such that
        we can find the buckets given delta(eps) and G:

            bucket_distribution = min_{bucket_distr} G.dot(bucket_distr) - [ delta(eps) for eps in eps_vector]

        with bucket_distr_i > 0 for all i. Please note that this method may lead to numerical errors with a large
        number_of_buckets or small amounts of probability mass in single buckets.

        [1] Sommer, David M., Sebastian Meiser, and Esfandiar Mohammadi. "Privacy loss classes: The central limit
            theorem in differential privacy." PoPETS 2019.2 (2019)
        """
        import scipy

        self.bucket_distribution = np.zeros(self.number_of_buckets, dtype=np.float64)
        self.error_correction = False

        # the eps vector we use for delta generation. We should have one value within
        eps_vec = ( np.arange(self.number_of_buckets + 1) - self.number_of_buckets // 2 - 1) * self.log_factor

        # The deltas we search to mimick with our future distribution.
        delta_vec = [ delta_func(eps) for eps in eps_vec ]

        # Generating G, the coefficient matrix for which we try to solve min_w  G.dot(w) - delta_func(eps_vec)
        try:
            if DP_type == 'adp':  # assuming delta_func describes approximate differential privacy
                # y: our base points (right edge of each bucket)
                y = ( np.arange(self.number_of_buckets) - self.number_of_buckets // 2 ) * self.log_factor

                y = y.reshape((1, len(y)))
                eps_vec = eps_vec.reshape((len(eps_vec), 1))

                G = np.maximum(0, 1 - np.exp(eps_vec - y))
                G = np.append(G, np.ones((self.number_of_buckets + 1, 1)), axis=1)  # for the infty bucket

            elif DP_type == 'pdp':  # assuming delta_func describes probabilistic differential privacy
                G = np.ones((self.number_of_buckets + 1, self.number_of_buckets + 1))
                G[np.tril_indices(self.number_of_buckets + 1, -1)] = 0

            else:
                raise NotImplementedError("DP_type '{}' not implemented.".format(DP_type))

        except MemoryError as e:
            raise MemoryError("Insufficient memory. Use smaller number_of_buckets.") from e

        # we try to solve min_w  G.dot(w) - delta_func(eps_vec) with w_i > 0 forall i
        w = scipy.optimize.nnls(G, delta_vec, maxiter=10 * G.shape[1])[0]

        self.bucket_distribution = w[:-1].copy()

        self.infty_bucket = w[-1]

        # distinguishing_events is zero as we cannot distinguish between infty bucket and distinguishing events
        self.distinguishing_events = np.float64(0.0)

        # and some internal stuff
        self.one_index = int(self.number_of_buckets // 2)   # this is a shortcut to the 0-bucket where L_A/B ~ 1
        self.u = np.int64(1)   # for error correction. Actually not needed
