import numpy as np
from operator import itemgetter
from dependency.dep_parsing import *
from collections import defaultdict, Counter


def get_trees(results):
    edges = []
    for result in results:
        line, tokenized_text, matrix_as_list = result
        directed_gold_edges = [(x.id, x.head) for x in line]
        edges.append(directed_gold_edges)
    return edges
        
        
# Compare parse obtained from perturbed masking to gold standard
def compare_parses(parse_args):
    parses, results, _ = decoding(parse_args)
    gold_parses = get_trees(results)
    tokens = [r[0] for r in results]
    assert len(parses) == len(gold_parses) == len(tokens)
    
    info = []
    parses_new = []
    tokens_new = []
    gold_parses_new = []
    
    sent_id = 0
      
    for parse, gold, token_list in zip(parses, gold_parses, tokens):
        assert len(parse) == len(gold) == len(token_list)
        
        if parse[-1][0] != gold[-1][0]: # DISCARD THESE (ID INCOMPATIBILITY)
            continue
        
        parses_new.append(parse)
        gold_parses_new.append(gold)
        tokens_new.append(token_list)
        
        for dep, gold_dep, token in zip(parse, gold, token_list):
            
            if dep == (0, -1) and gold_dep == (0, 0):
                continue
            
            head_id, gold_head_id = dep[1], gold_dep[1]
            head, gold_head = token_list[head_id], token_list[gold_head_id]
            
            info.append((sent_id, token, head, gold_head))
            
        sent_id += 1
    
    return info, tokens_new, parses_new, gold_parses_new    


# Get deprel-shift stats
def comparison_stats(comparisons, all_marker='ALL'):
    total = defaultdict(lambda:defaultdict(lambda:Counter()))
    errors = defaultdict(lambda:defaultdict(lambda:Counter()))
    error_sent_ids = defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
    distances = {'total':[], 'gold':[], 'error':[], 'correct':[]}
    
    for l in comparisons:
        sent_id, token, head, gold_head = l
        
        if token.deprel == '-root-':
            token.deprel = 'root'
        if head.deprel == '-root-':
            head.deprel = 'root'
        if gold_head.deprel == '-root-':
            gold_head.deprel = 'root'
        
        total[token.deprel][gold_head.deprel][head.deprel] += 1
        total[all_marker][gold_head.deprel][head.deprel] += 1
        total[token.deprel][all_marker][head.deprel] += 1
        total[token.deprel][gold_head.deprel][all_marker] += 1
        total[all_marker][all_marker][head.deprel] += 1
        total[all_marker][gold_head.deprel][all_marker] += 1
        total[token.deprel][all_marker][all_marker] += 1
        total[all_marker][all_marker][all_marker] += 1
        
        distances['total'].append(abs(head.id-token.id))
        distances['gold'].append(abs(gold_head.id-token.id))
        
        if head != gold_head:
            errors[token.deprel][gold_head.deprel][head.deprel] += 1
            errors[all_marker][gold_head.deprel][head.deprel] += 1
            errors[token.deprel][all_marker][head.deprel] += 1
            errors[token.deprel][gold_head.deprel][all_marker] += 1
            errors[all_marker][all_marker][head.deprel] += 1
            errors[all_marker][gold_head.deprel][all_marker] += 1
            errors[token.deprel][all_marker][all_marker] += 1
            errors[all_marker][all_marker][all_marker] += 1

            error_sent_ids[token.deprel][gold_head.deprel][head.deprel].append(sent_id)
            error_sent_ids[all_marker][gold_head.deprel][head.deprel].append(sent_id)
            error_sent_ids[token.deprel][all_marker][head.deprel].append(sent_id)
            error_sent_ids[token.deprel][gold_head.deprel][all_marker].append(sent_id)
            error_sent_ids[all_marker][all_marker][head.deprel].append(sent_id)
            error_sent_ids[all_marker][gold_head.deprel][all_marker].append(sent_id)
            error_sent_ids[token.deprel][all_marker][all_marker].append(sent_id)
            error_sent_ids[all_marker][all_marker][all_marker].append(sent_id)
            
            distances['error'].append(abs(head.id-token.id))
            
        else:
            distances['correct'].append(abs(head.id-token.id))
            
    errors_prc = defaultdict(lambda:defaultdict(lambda:Counter()))
    
    for d in errors:
        for g in errors[d]:
            for h in errors[d][g]:
                errors_prc[d][g][h] = errors[d][g][h] / total[d][g][h]
    
    distances_avg = {'total': np.average(distances['total']),
                     'gold': np.average(distances['gold']),
                     'error': np.average(distances['error']),
                     'correct': np.average(distances['correct'])}
    
    return total, errors, errors_prc, error_sent_ids, distances_avg


# Print stats
def error_stats(comparisons,
                all_marker='ALL',
                dep_list=[],
                gold_list=[],
                head_list=[],
                separator='-',
                min_abs=1,
                min_prc=0,
                abs_first=True,
                print_sent_ids=False,
                max_sent_ids=5,
                max_print=None):
    
    total, errors, errors_prc, error_sent_ids, distances_avg = comparison_stats(comparisons, all_marker=all_marker)
    
    stats = []
    sent_ids = {}
    
    deps = dep_list if dep_list else [x for x in errors if x != all_marker]
    for d in deps:
        
        golds = gold_list if gold_list else [x for x in errors[d] if x != all_marker]
        for g in golds:
            
            heads = head_list if head_list else [x for x in errors[d][g] if x != all_marker]
            for h in heads:
                
                err_abs = errors[d][g][h]
                err_prc = errors_prc[d][g][h]
                
                if err_abs >= min_abs and err_prc >= min_prc:
                    k = separator.join([d, g, h])
                    stats.append((k, err_prc, err_abs))
                
                    if print_sent_ids:
                        sent_ids[k] = error_sent_ids[d][g][h][:max_sent_ids]
    
    if abs_first:
        stats.sort(key=itemgetter(1), reverse=True)
        stats.sort(key=itemgetter(2), reverse=True)
    else:
        stats.sort(key=itemgetter(2), reverse=True)
        stats.sort(key=itemgetter(1), reverse=True)
    
    for k, err_prc, err_abs in stats[:max_print]:
        ids = sent_ids[k] if print_sent_ids else None
        k = k.split(separator)
        
        if ids:
            print(k[0], k[1], k[2], round(err_prc, 4), err_abs, ids, sep=' & ', end=' \\\\\n')
        else:
            print(k[0], k[1], k[2], round(err_prc, 4), err_abs, sep=' & ', end=' \\\\\n')


# Print specific examples
def print_change(ix, tokens, trees, gold_trees):
    for t, head, gold_head in zip(tokens[ix][1:], trees[ix][1:], gold_trees[ix][1:]):
        old = '\tCORR'
        new = ''
        if head != gold_head:
            old = 'OLD:' + '|'.join([str(tokens[ix][gold_head[1]].id), tokens[ix][gold_head[1]].form, tokens[ix][gold_head[1]].deprel])
            new = 'NEW:' + '|'.join([str(tokens[ix][head[1]].id), tokens[ix][head[1]].form, tokens[ix][head[1]].deprel])
        print(t, old, new, sep='\t')


parser = argparse.ArgumentParser()

# Data args
parser.add_argument('--matrix', default='results/bert-dist-PUD-12.pkl')

# Decoding args
parser.add_argument('--decoder', default='eisner')
parser.add_argument('--root', default='gold', help='gold or cls')
parser.add_argument('--subword', default='first')

args = parser.parse_args()

comp, tokens, trees, gold_trees = compare_parses(args)
total, errors, error_prc, error_ids, dist = comparison_stats(comp)
