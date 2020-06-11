#!/usr/local/bin/python

"""
Inter-annotator agreement calculator.

To use this script:
python iaa.py -t # folder1/ folder2/

 - # is the track number (1 - cohort selection or 2 - ADE)
 - folder1 contains the gold standard annotations
 - folder2 contains the test annotations

Please note that for Track 1, each file must contain 13 annotations,
one of each of the following tags:
   ABDOMINAL
   ADVANCED-CAD
   ALCOHOL-ABUSE
   ASP-FOR-MI
   CREATININE
   DIETSUPP-2MOS
   DRUG-ABUSE
   ENGLISH
   HBA1C
   KETO-1YR
   MAJOR-DIABETES
   MAKES-DECISIONS
   MI-6MOS

The file names in the annotations folder must match those in the
gold standard folder.

"""

import argparse
import glob
import os
from collections import defaultdict
from xml.etree import cElementTree


class ClinicalCriteria(object):
    """Criteria in the Track 1 documents."""

    def __init__(self, tid, value):
        """Init."""
        self.tid = tid.strip().upper()
        self.ttype = self.tid
        self.value = value.lower().strip()

    def equals(self, other, mode='strict'):
        """Return whether the current criteria is equal to the one provided."""
        if other.tid == self.tid and other.value == self.value:
            return True
        return False


class ClinicalConcept(object):
    """Named Entity Tag class."""

    def __init__(self, tid, start, end, ttype, text=''):
        """Init."""
        self.tid = str(tid).strip()
        self.start = int(start)
        self.end = int(end)
        self.text = str(text).strip()
        self.ttype = str(ttype).strip()

    def equals(self, other, mode='strict'):
        """Return whether the current tag is equal to the one provided."""
        assert mode in ('strict', 'lenient')
        if other.ttype == self.ttype:
            self_chars = set(range(self.start, self.end))
            othe_chars = set(range(other.start, other.end))
            if mode == 'strict':
                if self_chars == othe_chars:
                    return True
            else:   # lenient
                if self_chars & othe_chars:
                    return True
        return False


class Relation(object):
    """Relation class."""

    def __init__(self, rid, arg1, arg2, rtype):
        """Init."""
        assert isinstance(arg1, ClinicalConcept)
        assert isinstance(arg2, ClinicalConcept)
        self.rid = str(rid).strip()
        self.arg1 = arg1
        self.arg2 = arg2
        self.rtype = str(rtype).strip()

    def equals(self, other, mode='strict'):
        """Return whether the current tag is equal to the one provided."""
        assert mode in ('strict', 'lenient')
        if self.arg1.equals(other.arg1, mode) and \
                self.arg2.equals(other.arg2, mode) and \
                self.rtype == other.rtype:
            return True
        return False


class RecordTrack1(object):
    """Record for Track 2 class."""

    def __init__(self, file_path):
        self.path = os.path.abspath(file_path)
        self.basename = os.path.basename(self.path)
        self.annotations = self._get_annotations()
        self.text = None

    @property
    def tags(self):
        return self.annotations['tags']

    def _get_annotations(self):
        """Return a dictionary with all the annotations in the .ann file."""
        annotations = defaultdict(dict)
        annotation_file = cElementTree.parse(self.path)
        for tag in annotation_file.findall('.//TAGS/*'):
            annotations['tags'][tag.tag.upper()] = ClinicalCriteria(
                tag.tag.upper(), tag.attrib['met'])
        return annotations


class RecordTrack2(object):
    """Record for Track 2 class."""

    def __init__(self, file_path):
        """Initialize."""
        self.path = os.path.abspath(file_path)
        self.basename = os.path.basename(self.path)
        self.annotations = self._get_annotations()
        self.text = self._get_text()

    @property
    def tags(self):
        return self.annotations['tags']

    @property
    def relations(self):
        return self.annotations['relations']

    def _get_annotations(self):
        """Return a dictionary with all the annotations in the .ann file."""
        annotations = defaultdict(dict)
        with open(self.path) as annotation_file:
            lines = annotation_file.readlines()
            for line_num, line in enumerate(lines):
                if line.strip().startswith('T'):
                    tag_id, tag_m, tag_text = line.strip().split('\t')
                    if len(tag_m.split(' ')) == 3:
                        tag_type, tag_start, tag_end = tag_m.split(' ')
                    elif len(tag_m.split(' ')) == 4:
                        tag_type, tag_start, _, tag_end = tag_m.split(' ')
                    tag_start, tag_end = int(tag_start), int(tag_end)
                    annotations['tags'][tag_id] = ClinicalConcept(tag_id,
                                                                  tag_start,
                                                                  tag_end,
                                                                  tag_type,
                                                                  tag_text)
            for line_num, line in enumerate(lines):
                if line.strip().startswith('R'):
                    rel_id, rel_m = line.strip().split('\t')
                    rel_type, rel_arg1, rel_arg2 = rel_m.split(' ')
                    rel_arg1 = rel_arg1.split(':')[1]
                    rel_arg2 = rel_arg2.split(':')[1]
                    arg1 = annotations['tags'][rel_arg1]
                    arg2 = annotations['tags'][rel_arg2]
                    annotations['relations'][rel_id] = Relation(rel_id, arg1,
                                                                arg2, rel_type)
        return annotations

    def _get_text(self):
        """Return the text in the corresponding txt file."""
        path = self.path.replace('.ann', '.txt')
        with open(path) as text_file:
            text = text_file.read()
        return text

    def search_by_id(self, key):
        """Search by id among both tags and relations."""
        try:
            return self.annotations['tags'][key]
        except KeyError():
            try:
                return self.annotations['relations'][key]
            except KeyError():
                return None


class Evaluator(object):
    """Abstract methods and var to evaluate."""

    def __init__(self):
        """Initialise."""
        self.scores = {'tags': {'tp': 0, 'fp': 0, 'fn': 0},
                       'relations': {'tp': 0, 'fp': 0, 'fn': 0}}

    def precision(self, target='tags'):
        """Compute Precision score."""
        scores = self.scores.get(target)
        try:
            return scores['tp'] / (scores['tp'] + scores['fp'])
        except ZeroDivisionError:
            return 0.0

    def recall(self, target='tags'):
        """Compute Recall score."""
        assert target in ('tags', 'relations')
        scores = self.scores.get(target)
        try:
            return scores['tp'] / (scores['tp'] + scores['fn'])
        except ZeroDivisionError:
            return 0.0

    def f_score(self, beta=1, target='tags'):
        """Compute F1-measure score."""
        assert target in ('tags', 'relations')
        assert beta > 0.
        try:
            num = (1 + beta**2) * (self.precision(target) * self.recall(target))
            den = beta**2 * (self.precision(target) + self.recall(target))
            return num / den
        except ZeroDivisionError:
            return 0.0

    def f1(self, target='tags'):
        """Compute the F1-score (beta=1)."""
        assert target in ('tags', 'relations')
        return self.f_score(beta=1, target=target)


class SingleEvaluator(Evaluator):
    """Evaluate two single files."""

    def __init__(self, doc1, doc2, track, mode='strict', key=None):
        """Initialize."""
        assert isinstance(doc1, RecordTrack2) or isinstance(doc1, RecordTrack1)
        assert isinstance(doc2, RecordTrack2) or isinstance(doc2, RecordTrack1)
        assert mode in ('strict', 'lenient')
        assert doc1.basename == doc2.basename
        super(SingleEvaluator, self).__init__()
        self.doc1 = doc1
        self.doc2 = doc2
        if key:
            gol = [t for t in doc1.tags.values() if t.ttype == key]
            sys = [t for t in doc2.tags.values() if t.ttype == key]
        else:
            gol = [t for t in doc1.tags.values()]
            sys = [t for t in doc2.tags.values()]
        self.scores['tags']['tp'] = len({s.tid for s in sys for g in gol if g.equals(s, mode)})
        self.scores['tags']['fp'] = len({s.tid for s in sys}) - self.scores['tags']['tp']
        self.scores['tags']['fn'] = len({g.tid for g in gol}) - len({s.tid for s in sys})
        if track == 2:
            if key:
                gol = [r for r in doc1.relations.values() if r.rtype == key]
                sys = [r for r in doc2.relations.values() if r.rtype == key]
            else:
                gol = [r for r in doc1.relations.values()]
                sys = [r for r in doc2.relations.values()]
            self.scores['relations']['tp'] = len({s.rid for s in sys for g in gol if g.equals(s, mode)})
            self.scores['relations']['fp'] = len({s.rid for s in sys}) - self.scores['relations']['tp']
            self.scores['relations']['fn'] = len({g.rid for g in gol}) - len({s.rid for s in sys})


class MultipleEvaluator(Evaluator):
    """Evaluate two sets of files."""

    def __init__(self, f1, f2, track, files, tag_type=None, mode='strict'):
        """Initialize."""
        assert os.path.exists(f1)
        assert os.path.exists(f2)
        assert type(files) in (list, tuple, set)
        assert mode in ('strict', 'lenient')
        super(MultipleEvaluator, self).__init__()
        if track == 1:
            self.tags = ('ABDOMINAL', 'ADVANCED-CAD', 'ALCOHOL-ABUSE',
                         'ASP-FOR-MI', 'CREATININE', 'DIETSUPP-2MOS',
                         'DRUG-ABUSE', 'ENGLISH', 'HBA1C', 'KETO-1YR',
                         'MAJOR-DIABETES', 'MAKES-DECISIONS', 'MI-6MOS')
        else:
            self.tags = ('Drug', 'Strength', 'Duration', 'Route', 'Form',
                         'ADE', 'Dosage', 'Reason', 'Frequency')
        self.relations = ('Strength-Drug', 'Dosage-Drug', 'Duration-Drug',
                          'Frequency-Drug', 'Form-Drug', 'Route-Drug',
                          'Reason-Drug', 'ADE-Drug')
        self.scores = {'tags': {'tp': 0, 'fp': 0, 'fn': 0},
                       'relations': {'tp': 0, 'fp': 0, 'fn': 0}}

        self.scores['tags']['micro'] = {'precision': 0, 'recall': 0, 'f1': 0}
        self.scores['tags']['macro'] = {'precision': 0, 'recall': 0, 'f1': 0}
        self.scores['relations']['micro'] = {'precision': 0, 'recall': 0, 'f1': 0}
        self.scores['relations']['macro'] = {'precision': 0, 'recall': 0, 'f1': 0}

        for file in files:
            if track == 1:
                g = RecordTrack1(os.path.join(f1, file))
                s = RecordTrack1(os.path.join(f2, file))
            else:
                g = RecordTrack2(os.path.join(f1, file))
                s = RecordTrack2(os.path.join(f2, file))
            evaluator = SingleEvaluator(g, s, track, mode, tag_type)
            for target in ('tags', 'relations'):
                for score in ('tp', 'fp', 'fn'):
                    self.scores[target][score] += evaluator.scores[target][score]
                for score in ('precision', 'recall', 'f1'):
                    fn = getattr(evaluator, score)
                    self.scores[target]['macro'][score] += fn(target)

        for target in ('tags', 'relations'):
            # Normalization
            for key in self.scores[target]['macro'].keys():
                self.scores[target]['macro'][key] = \
                    self.scores[target]['macro'][key] / len(files)

            for key in self.scores[target]['micro'].keys():
                fn = getattr(self, key)
                self.scores[target]['micro'][key] = fn(target)


def evaluate(f1, f2, track, files, mode='strict'):
    """Run the evaluation by considering only files in the two folders."""
    assert mode in ('strict', 'lenient')
    evaluator_s = MultipleEvaluator(f1, f2, track, files)

    if track == 1:
        print('{:*^28}'.format(' CRITERIA '))
        print('{:20}  {:6}'.format('', 'Acc.'))
        for tag in evaluator_s.tags:
            evaluator_tag_s = MultipleEvaluator(f1, f2, track, files, tag)
            print('{:>20}  {:<5.4f}'.format(
                tag.capitalize(),
                evaluator_tag_s.scores['tags']['micro']['precision']))
        print('{:>20}  {:-^6}'.format('', ''))
        print('{:>20}  {:<5.4f}'.format(
            'Overall',
            evaluator_s.scores['tags']['micro']['precision']))
        print()
        print('{:^28}'.format('  {} files found  '.format(len(files))))
    else:
        evaluator_l = MultipleEvaluator(f1, f2, track, files, mode='lenient')
        print('{:*^70}'.format(' NAMED ENTITIES '))
        print('{:20}  {:-^22}    {:-^22}'.format('', ' strict ', ' lenient '))
        print('{:20}  {:6}  {:6}  {:6}    {:6}  {:6}  {:6}'.format('', 'Prec.',
                                                                   'Rec.',
                                                                   'F(b=1)',
                                                                   'Prec.',
                                                                   'Rec.',
                                                                   'F(b=1)'))
        for tag in evaluator_s.tags:
            evaluator_tag_s = MultipleEvaluator(f1, f2, track, files, tag)
            evaluator_tag_l = MultipleEvaluator(f1, f2, track, files, tag, mode='lenient')
            print('{:>20}  {:<5.4f}  {:<5.4f}  {:<5.4f}    {:<5.4f}  {:<5.4f}  {:<5.4f}'.format(
                tag.capitalize(),
                evaluator_tag_s.scores['tags']['micro']['precision'],
                evaluator_tag_s.scores['tags']['micro']['recall'],
                evaluator_tag_s.scores['tags']['micro']['f1'],
                evaluator_tag_l.scores['tags']['micro']['precision'],
                evaluator_tag_l.scores['tags']['micro']['recall'],
                evaluator_tag_l.scores['tags']['micro']['f1']))
        print('{:>20}  {:-^48}'.format('', ''))
        print('{:>20}  {:<5.4f}  {:<5.4f}  {:<5.4f}    {:<5.4f}  {:<5.4f}  {:<5.4f}'.format(
            'Overall (micro)',
            evaluator_s.scores['tags']['micro']['precision'],
            evaluator_s.scores['tags']['micro']['recall'],
            evaluator_s.scores['tags']['micro']['f1'],
            evaluator_l.scores['tags']['micro']['precision'],
            evaluator_l.scores['tags']['micro']['recall'],
            evaluator_l.scores['tags']['micro']['f1']))
        print('{:>20}  {:<5.4f}  {:<5.4f}  {:<5.4f}    {:<5.4f}  {:<5.4f}  {:<5.4f}'.format(
            'Overall (macro)',
            evaluator_s.scores['tags']['macro']['precision'],
            evaluator_s.scores['tags']['macro']['recall'],
            evaluator_s.scores['tags']['macro']['f1'],
            evaluator_l.scores['tags']['macro']['precision'],
            evaluator_l.scores['tags']['macro']['recall'],
            evaluator_l.scores['tags']['macro']['f1']))
        print()

        print('{:*^70}'.format(' RELATIONS '))
        for rel in evaluator_s.relations:
            evaluator_tag_s = MultipleEvaluator(f1, f2, track, files, rel, mode='strict',)
            evaluator_tag_l = MultipleEvaluator(f1, f2, track, files, rel, mode='lenient')
            print('{:>20}  {:<5.4f}  {:<5.4f}  {:<5.4f}    {:<5.4f}  {:<5.4f}  {:<5.4f}'.format(
                '{} -> {}'.format(rel.split('-')[0], rel.split('-')[1].capitalize()),
                evaluator_tag_s.scores['relations']['micro']['precision'],
                evaluator_tag_s.scores['relations']['micro']['recall'],
                evaluator_tag_s.scores['relations']['micro']['f1'],
                evaluator_tag_l.scores['relations']['micro']['precision'],
                evaluator_tag_l.scores['relations']['micro']['recall'],
                evaluator_tag_l.scores['relations']['micro']['f1']))
        print('{:>20}  {:-^48}'.format('', ''))
        print('{:>20}  {:<5.4f}  {:<5.4f}  {:<5.4f}    {:<5.4f}  {:<5.4f}  {:<5.4f}'.format(
            'Overall (micro)',
            evaluator_s.scores['relations']['micro']['precision'],
            evaluator_s.scores['relations']['micro']['recall'],
            evaluator_s.scores['relations']['micro']['f1'],
            evaluator_l.scores['relations']['micro']['precision'],
            evaluator_l.scores['relations']['micro']['recall'],
            evaluator_l.scores['relations']['micro']['f1']))
        print('{:>20}  {:<5.4f}  {:<5.4f}  {:<5.4f}    {:<5.4f}  {:<5.4f}  {:<5.4f}'.format(
            'Overall (macro)',
            evaluator_s.scores['relations']['macro']['precision'],
            evaluator_s.scores['relations']['macro']['recall'],
            evaluator_s.scores['relations']['macro']['f1'],
            evaluator_l.scores['relations']['macro']['precision'],
            evaluator_l.scores['relations']['macro']['recall'],
            evaluator_l.scores['relations']['macro']['f1']))
        print()
        print('{:20}{:^48}'.format('', '  {} files found  '.format(len(files))))


def main(f1, f2, track):
    """Where the magic begins."""
    # Select the .ann files existing in both folders.
    if track == 1:
        file_ext = '*.xml'
    else:
        file_ext = '*.ann'
    folde1 = os.path.join(f1, file_ext)
    files1 = set([os.path.basename(f) for f in glob.glob(folde1)])
    folde2 = os.path.join(f1, file_ext)
    files2 = set([os.path.basename(f) for f in glob.glob(folde2)])
    files = files1 & files2
    if files1 - files:
        print('Files skipped in {}:'.format(os.path.basename(f1)))
        for f in sorted(list(files1 - files)):
            print(' - {}'.format(f))
    if files2 - files:
        print('Files skipped in {}:'.format(os.path.basename(f2)))
        for f in sorted(list(files2 - files)):
            print(' - {}'.format(f))
    evaluate(f1, f2, track, files)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='n2c2: Evaluation script')
    parser.add_argument('-t', '--track', choices={'1', '2'}, help='Track number')
    parser.add_argument('folder1', help='First data folder path')
    parser.add_argument('folder2', help='Second data folder path')
    args = parser.parse_args()
    if args.folder1 and args.folder2 and args.track:
        main(os.path.abspath(args.folder1), os.path.abspath(args.folder2),
         int(args.track))
    else:
        print("To use this script:\n python iaa.py -t # folder1/ folder2/\n"+
              "where # is the track number (1 - cohort selection or 2 - ADE)\n"+
              " folder1 contains the gold standard annotations \n"+
              " folder2 contains the test annotations")
