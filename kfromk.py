import argparse

import numpy
import scipy.cluster.vq

import echonest.remix.audio as audio

def main(afilename, bfilename, outfilename, k=8):

    Kfromk(afilename, bfilename,outfilename).run(mix=1.0, envelope=False).render(outfilename)


def cluster_segs(segs, feature_getter=lambda x: x.timbre, k=8):
    # whiten segs???
    seg_features = numpy.array([feature_getter(x) for x in segs])
    code_book = scipy.cluster.vq.kmeans(seg_features, k)
    indices, dists = scipy.cluster.vq.vq(seg_features, code_book[0])
    clusters = [ [] for x in range(k) ]
    for i in range(len(indices)):
        # how well does this fit into the cluster?
        segs[i].dist = int((dists[i] - min(dists)) * 255.0 / (max(dists) - min(dists)))
        clusters[indices[i]].append(segs[i])
    return clusters

def sort_clusters(clusters):
    for i in range(len(clusters)):
        clusters[i] = sorted(clusters[i], key=lambda x: x.dist, reverse=False)
    return clusters

def get_indices(segs, feature_getter=lambda x: x.timbre, k=8):
    # TODO: factor out code shared w cluster_segs

    seg_features = numpy.array([feature_getter(x) for x in segs])

    code_book = scipy.cluster.vq.kmeans(seg_features, k)
    indices, dists = scipy.cluster.vq.vq(seg_features, code_book[0])
    clusters = [ [] for x in range(k) ]
    for i in range(len(indices)):
        # how well does this fit into the cluster?
        segs[i].dist = int((dists[i] - min(dists)) * 255.0 / (max(dists) - min(dists)))
        clusters[indices[i]].append(segs[i])
    
    clusters = sort_clusters(clusters)
    cluster_lengths = [len(x) for x in clusters]

    seg_cluster_indices = []
    for i in range(len(indices)):
        cluster_index = indices[i]
        index_in_cluster = clusters[cluster_index].index(segs[i])
        relative_index_in_cluster = float(index_in_cluster) / cluster_lengths[cluster_index]

        seg_cluster_indices.append(
            (indices[i], index_in_cluster, relative_index_in_cluster)
            )

    return seg_cluster_indices


class Kfromk(object):

    def __init__(self, afilename, bfilename, outfilename):
        self.input_a = audio.LocalAudioFile(afilename)
        self.input_b = audio.LocalAudioFile(bfilename)
        self.segs_a = self.input_a.analysis.segments
        self.segs_b = self.input_b.analysis.segments
        self.outfilename = outfilename


    def run(self, mix=0.5, envelope=False):

        self.compute_clusters()

        self.out = self.initialize_output()

        self.match_segs(mix, envelope)

        return self
    
    def compute_clusters(self):

        self.seg_cluster_indices_a = get_indices(self.segs_a)

        self.clusters_b = sort_clusters(cluster_segs(self.segs_b))


    def match_segs(self, mix, envelope):
        for i, a_seg in enumerate(self.segs_a):

            # find the best match
            match_seg = self.find_match(a_seg)
            segment_data = self.input_b[match_seg]
            reference_data = self.input_a[a_seg]

            # fix segment length: if new seg is shorter, add silence
            # if new seg is longer, truncate it
            segment_data = self.correct_segment_length(segment_data, reference_data)
            
            # apply the volume envelope from each seg in A to the matching seg in B
            if envelope:
                segment_data = self.apply_envelope(a_seg, segment_data)

            # mix the seg from B with the seg from A
            mixed_data = audio.mix(segment_data, reference_data, mix=mix)
            
            # self.out.append(mixed_data)
            next_time = a_seg.start
            self.out.add_at(next_time, mixed_data)

    def initialize_output(self):
        # This chunk creates a new array of AudioData to put the resulting resynthesis in:

        # Add two seconds to the length, just in case
        dur = len(self.input_a.data) + 100000 

        # This determines the 'shape' of new array.
        # (Shape is a tuple (x, y) that indicates the length per channel of the audio file)
        # If we have a stereo shape, copy that shape
        if len(self.input_a.data.shape) > 1:
            new_shape = (dur, self.input_a.data.shape[1])
            new_channels = self.input_a.data.shape[1]
        # If not, make a mono shape
        else:
            new_shape = (dur,)
            new_channels = 1
        # This creates the new AudioData array, based on the new shape
        out = audio.AudioData(shape=new_shape,
                            sampleRate=self.input_b.sampleRate,
                            numChannels=new_channels)
        return out

    def find_match(self, a_seg):
        seg_index = a_seg.absolute_context()[0]

        seg_indices = self.seg_cluster_indices_a[seg_index]

        cluster_index, seg_in_cluster_index, relative_index = seg_indices

        b_len = len(self.clusters_b[cluster_index])
        index_in_cluster_b = int(numpy.math.floor(relative_index * b_len))

        b_seg = self.clusters_b[cluster_index][index_in_cluster_b]

        return b_seg

    def correct_segment_length(self, segment_data, reference_data):
        # if new segment is too short, pad w silence
        if segment_data.endindex < reference_data.endindex:
            if self.out.numChannels > 1:
                silence_shape = (reference_data.endindex, self.out.numChannels)
            else:
                silence_shape = (reference_data.endindex,)
            new_segment = audio.AudioData(shape=silence_shape,
                                    sampleRate=self.out.sampleRate,
                                    numChannels=segment_data.numChannels)
            new_segment.append(segment_data)
            new_segment.endindex = len(new_segment)

        # or if new segment is too long, truncate it
        elif segment_data.endindex > reference_data.endindex:
            new_segment = segment_data
            # index = slice(0, int(reference_data.endindex), 1)
            # new_segment = audio.AudioData(None,segment_data.data[index],
            #                         sampleRate=segment_data.sampleRate)

        return new_segment

    def apply_envelope(self, a_seg, segment_data):
        seg_index = a_seg.absolute_context()[0]

        # This gets the maximum volume and starting volume for the segment from A:
        # db -> voltage ratio http://www.mogami.com/e/cad/db.html
        linear_max_volume = pow(10.0,a_seg.loudness_max/20.0)
        linear_start_volume = pow(10.0,a_seg.loudness_begin/20.0)

        # This gets the starting volume for the next segment
        if(seg_index == len(self.segs_a)-1): # If this is the last segment, the next volume is zero
            linear_next_start_volume = 0
        else:
            linear_next_start_volume = pow(10.0,self.segs_a[seg_index+1].loudness_begin/20.0)
            pass

        # This gets when the maximum volume occurs in A
        when_max_volume = a_seg.time_loudness_max

        # Count # of ticks I wait doing volume ramp so I can fix up rounding errors later.
        ss = 0
        # This sets the starting volume volume of this segment. 
        cur_vol = float(linear_start_volume)
        # This  ramps up to the maximum volume from start
        samps_to_max_loudness_from_here = int(segment_data.sampleRate * when_max_volume)
        if(samps_to_max_loudness_from_here > 0):
            how_much_volume_to_increase_per_samp = float(linear_max_volume - linear_start_volume)/float(samps_to_max_loudness_from_here)
            for samps in xrange(samps_to_max_loudness_from_here):
                try:
                    # This actally applies the volume modification
                    segment_data.data[ss] *= cur_vol
                except IndexError:
                    pass
                cur_vol = cur_vol + how_much_volume_to_increase_per_samp
                ss = ss + 1
        # This ramp down to the volume for the start of the next segent
        samps_to_next_segment_from_here = int(segment_data.sampleRate * (a_seg.duration-when_max_volume))
        if(samps_to_next_segment_from_here > 0):
            how_much_volume_to_decrease_per_samp = float(linear_max_volume - linear_next_start_volume)/float(samps_to_next_segment_from_here)
            for samps in xrange(samps_to_next_segment_from_here):
                cur_vol = cur_vol - how_much_volume_to_decrease_per_samp
                try:
                    # This actally applies the volume modification
                    segment_data.data[ss] *= cur_vol
                except IndexError:
                    pass
                ss = ss + 1
    
        return segment_data

    def render(self, output_filename):
        # This writes the newly created audio to the given file.  Phew!
        self.out.encode(output_filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", help="re-synthesize this track", type=str, required=True)
    parser.add_argument("-b", help="use the segments of this track for re-synthesis", type=str, required=True)
    parser.add_argument("--output", "-o", help="where the output goes", type=str, required=True)
    parser.add_argument("-k", help="how many clusters", type=int, default=8)
    
    args = parser.parse_args()

    print args
    main(
        afilename=args.a, 
        bfilename=args.b,
        outfilename=args.output,
        k=args.k
        )
