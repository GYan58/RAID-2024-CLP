from Settings import *
from Utils import *

def MultiKrum(params, frac, num=1, uids=None):
    n = len(params)
    m = n - frac
    if m <= 1:
        m = n
    
    distances = defaultdict(dict)
    keys = params[0].keys()

    for i in range(n):
        param1 = params[i]
        for j in range(i, n):
            param2 = params[j]
            if i == j:
                distances[i][j] = 0.0
                continue
            distance = 0.0
            if j > i:
                for key in keys:
                    if "weight" in key or "bias" in key:
                        distance += torch.norm(param1[key] - param2[key]).item() ** 2
                distance = np.sqrt(distance)
                distances[i][j] = distance
                distances[j][i] = distance

    if num == 1:
        find_id, find_val = -1, 1e20
        for i in range(n):
            dist = sorted(distances[i].values())
            sum_dist = sum(dist[:m])
            if find_val > sum_dist:
                find_val = sum_dist
                find_id = i
        return params[find_id], [find_id]

    elif num >= 2:
        p_dict = {i: sum(sorted(val.values())[:m]) for i, val in distances.items()}
        s_dict = sorted(p_dict.items(), key=lambda x: x[1])

        good_params, good_ids = [], []
        for i in range(num):
            key = s_dict[i][0]
            good_ids.append(uids[key] if uids else key)
            good_params.append(params[key])
        bad_ids = sorted(set(range(n)) - set(good_ids))
        return avg_params(good_params), bad_ids

def TrimMean(params, frac):
    n = len(params)
    k = min(frac, int(n / 2) - 1)
    f_param = {key: torch.zeros_like(val) for key, val in params[0].items()}

    for key in f_param.keys():
        if "bias" in key or "weight" in key:
            all_params = torch.stack([params[i][key] for i in range(n)])
            sorted_params, _ = torch.sort(all_params, dim=0)
            trimmed_params = sorted_params[k:n - k]
            f_param[key] = torch.mean(trimmed_params, dim=0)
        else:
            f_param[key] = sum(params[i][key] for i in range(n)) / n

    return f_param, [-1, -1]

def Median(params, frac=None):
    n = len(params)

    f_param = {key: torch.zeros_like(val) for key, val in params[0].items()}
    C = 0
    for key in f_param.keys():
        if "bias" in key or "weight" in key:
            all_params = torch.stack([params[i][key] for i in range(n)])
            f_param[ky] = torch.median(all_params, dim=0)
        else:
            f_param[key] = sum(params[i][key] for i in range(n)) / n
    
    return f_param, [-1,-1]

class AFA:
    def __init__(self):
        self.alphas = {}
        self.betas = {}

    def add(self, id):
        self.alphas[id] = 0.5
        self.betas[id] = 0.5

    def agg_params(self, ids, r_param, params, lens):
        for key in ids:
            if key not in self.alphas:
                self.add(key)

        local_grads = {ids[i]: get_grad(params[i], r_param) for i in range(len(params))}
        lens = {ids[i]: lens[i] for i in range(len(params))}
        pks = {key: self.alphas[key] / (self.alphas[key] + self.betas[key]) for key in ids}

        good_ids = ids.copy()
        bad_ids, epi, step = [], 0.5, 1

        while True:
            grads = [local_grads[key] for key in good_ids]
            grad_lens = [lens[key] * pks[key] for key in good_ids]
            grad_r = avg_params(grads, grad_lens)

            sims = {key: get_sim(local_grads[key], grad_r) for key in good_ids}
            a_sims = list(sims.values())

            mu, std, med = np.mean(a_sims), np.std(a_sims), np.median(a_sims)
            remove_ids = []

            for key in good_ids:
                sim = sims[key]
                if (mu < med and sim < med - std * epi) or (mu >= med and sim > med + std * epi):
                    bad_ids.append(key)
                    remove_ids.append(key)

            if not remove_ids:
                break

            good_ids = list(set(good_ids) - set(remove_ids))
            epi += step
        
        if len(bad_ids) >= len(ids) / 2:
            good_ids = ids

        good_grads = [local_grads[key] for key in good_ids]
        good_lens = [lens[key] * pks[key] for key in good_ids]
        grad_res = avg_params(good_grads, good_lens)

        res = minus_params(r_param, -1, grad_res)
        avg_params_ = avg_params(params)

        for key in res.keys():
            if "bias" not in key and "weight" not in key:
                res[key] = avg_params_[key]

        for key in good_ids:
            self.alphas[key] += 1
        for key in bad_ids:
            self.betas[key] += 1

        return res, bad_ids

def cosDefense(last_param, params, lens, uids):
    grads = [minus_params(params[i], 1, last_param) for i in range(len(params))]
    
    avg_grad = last_param
    avg_grad_layer = extract_layer(avg_grad)[-1]
    
    e_layers = [extract_layer(grads[i])[-1] for i in range(len(params))]

    sims = [F.cosine_similarity(avg_grad_layer, e_layer, dim=1).item() for e_layer in e_layers]

    min_val, max_val = np.min(sims), np.max(sims)
    norm_sims = [(sim - min_val) / (max_val - min_val) for sim in sims]

    threshold = np.mean(norm_sims)
    bad_ids = [i for i in range(len(norm_sims)) if norm_sims[i] >= threshold]

    good_params = [params[i] for i in range(len(uids)) if i not in bad_ids]
    good_lens = [lens[i] for i in range(len(uids)) if i not in bad_ids]

    result = avg_params(good_params, good_lens)
    return result, bad_ids
