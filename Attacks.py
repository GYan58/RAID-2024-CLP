from Settings import *
from Utils import *

def attk_Fang(re_para, k_paras, e_paras, g_num, a_paras):
    if g_num <= 1:
        return [re_para]

    direction = get_directions(k_paras, re_para)
    avg_k_paras = avg_params(k_paras)

    goal_ids = []
    find_para = None
    find_lambda = 0.01
    stop = False
    count = 0
    while not stop:
        r_para = cp.deepcopy(re_para)
        n_para = minus_params(r_para, find_lambda, direction)
        attack_paras = [n_para] * g_num + e_paras

        _, id = AggMKrum(attack_paras, g_num)

        if id in range(g_num):
            stop = True
            find_para = n_para
        else:
            find_lambda *= 0.5

        if find_lambda < 0.000001:
            stop = True
            find_para = n_para
        count += 1
    
    return [find_para] * g_num


def attk_MinMax(re_para, k_paras, g_num, a_paras):
    if g_num <= 1:
        return [re_para]

    direction = get_directions(k_paras, re_para)
    avg_k_paras = avg_params(k_paras)
    grads = [get_grad(k_paras[i], re_para) for i in range(len(k_paras))]
    avg_grads = avg_params(grads)

    max_dist = max(get_dist(grads[i], grads[j]) for i in range(len(k_paras)) for j in range(i, len(k_paras)))

    gamma = 0.001
    stop = False
    find_grad = None
    while not stop:
        n_grad = minus_params(avg_grads, gamma, direction)
        max_dist_curr = max(get_dist(n_grad, grads[i]) for i in range(len(k_paras)))

        if max_dist_curr < max_dist or gamma < 0.000001:
            stop = True
            find_grad = n_grad
        else:
            gamma *= 0.5

    find_para = minus_params(re_para, -1, find_grad)
    return [find_para] * g_num


def attk_MinSum(re_para, k_paras, g_num, a_paras):
    if g_num <= 1:
        return [re_para]

    direction = get_directions(k_paras, re_para)
    avg_k_paras = avg_params(k_paras)
    grads = [get_grad(k_paras[i], re_para) for i in range(len(k_paras))]
    avg_grads = avg_params(grads)

    max_dist = max(sum(get_dist(grads[i], grads[j]) for j in range(len(k_paras))) for i in range(len(k_paras)))

    gamma = 0.001
    stop = False
    find_grad = None
    while not stop:
        n_grad = minus_params(avg_grads, gamma, direction)
        total_dist = sum(get_dist(n_grad, grads[i]) for i in range(len(k_paras)))

        if total_dist < max_dist or gamma < 0.000001:
            stop = True
            find_grad = n_grad
        else:
            gamma *= 0.5

    find_para = minus_params(re_para, -1, find_grad)
    return [find_para] * g_num


def attk_Lie(re_para, k_paras, g_num, a_num, a_paras):
    if g_num <= 1:
        return [re_para]

    n = g_num
    m = int(n * 0.25)
    s = int(n / 2 + 1) - m
    z = st.norm.ppf((n - m - s) / (n - m))

    grads = [get_grad(k_paras[i], re_para) for i in range(len(k_paras))]
    avg_grads = avg_params(grads)
    direction = get_directions(k_paras, re_para)

    f_grad = cp.deepcopy(avg_grads)
    keys = k_paras[0].keys()
    for key in keys:
        if "bias" in key or "weight" in key:
            grad_values = np.array([grads[i][key].cpu().detach().numpy() for i in range(len(grads))])
            mu = np.mean(grad_values, axis=0)
            std = np.std(grad_values, axis=0)
            dir_values = direction[key].cpu().detach().numpy()

            result = mu + z * std * (dir_values < 0) - z * std * (dir_values > 0)
            f_grad[key] = torch.from_numpy(result).to(device)

    find_para = minus_params(re_para, -1, f_grad)
    return [find_para] * g_num


class MPHM:
    def __init__(self):
        self.last_b_grad = None

    def get_sigma(self, grads):
        vecs = [np.concatenate([grad[key].cpu().detach().numpy().reshape(-1) for key in grad if "weight" in key or "bias" in key]) for grad in grads]
        sigmas = np.std(vecs, axis=0)
        norm = np.linalg.norm(sigmas)
        return norm

    def attk_mphm(self, re_para, k_paras, g_num, a_paras):
        if g_num <= 1:
            return [re_para]

        direction = get_directions(k_paras, re_para)
        avg_k_paras = avg_params(k_paras)
        grads = [get_grad(k_paras[i], re_para) for i in range(len(k_paras))]
        avg_grads = avg_params(grads)
        delta_avg_grads = minus_params(avg_grads, -0.5, self.last_b_grad) if self.last_b_grad else cp.deepcopy(avg_grads)

        sigma_norm = self.get_sigma(grads)
        grad_norm = get_norm(delta_avg_grads)
        lambda_val = sigma_norm / grad_norm
        find_grad = minus_params(avg_grads, lambda_val, delta_avg_grads)
        self.last_b_grad = cp.deepcopy(find_grad)

        find_para = minus_params(re_para, -1, find_grad)
        return [find_para] * g_num


def attack_GraSP(re_para, k_paras, a_paras, attack_value=0.1):
    if len(a_paras) <= 1:
        return [re_para]
    
    direction = get_directions(k_paras, re_para)

    avg_param = avg_params(k_paras)
    gradient = get_grad(avg_param, re_para)
    gradients = [get_grad(param, re_para) for param in a_paras]
    
    attack_degree = attack_value
    if attack_degree != 0:
        a = get_dot_product(gradient, direction) ** 2 - attack_degree ** 2 * get_norm(gradient) ** 2 * get_norm(direction) ** 2
        b = 2 * attack_degree ** 2 * get_norm(gradient) ** 2 * get_dot_product(gradient, direction) - 2 * get_dot_product(gradient, gradient) * get_dot_product(gradient, direction)
        c = -attack_degree ** 2 * get_norm(gradient) ** 4 + get_dot_product(gradient, gradient) ** 2
        s1 = (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
        s2 = (-b - np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
    
    if attack_degree > 0:
        upper_bound = s2
    elif attack_degree < 0:
        if s1 < 0:
            s1 = find_lambda(gradient, direction, attack_degree)
        upper_bound = s1
    else:
        upper_bound = get_norm(gradient) ** 2 / get_dot_product(gradient, direction)
    
    valid_count = 0
    total_similarity = 0
    initial_similarities = {}
    upper_bound = min(0.1, upper_bound)
    adjusted_gradient = minus_params(gradient, upper_bound, direction)
    
    for i in range(len(gradients)):
        similarity = get_dot_product(gradient, gradients[i]) / get_norm(gradient) / get_norm(adjusted_gradient)
        initial_similarities[i] = similarity
        if similarity > attack_value:
            valid_count += 1
        else:
            total_similarity += similarity

    if valid_count < len(initial_similarities):
        attack_degree = (attack_value * len(initial_similarities) - total_similarity) / valid_count

    lambdas = {}
    for i in range(len(gradients)):
        if initial_similarities[i] > attack_degree:
            lambda_value = (-attack_degree * get_norm(gradient) * get_norm(adjusted_gradient) + get_dot_product(gradient, gradients[i])) / get_dot_product(gradient, direction)
            lambdas[i] = lambda_value

    lambda_values = list(lambdas.values())
    find_para = []
    for i in range(len(lambdas)):
        adjusted_gradient = minus_params(gradients[i], lambdas[i], direction)
        poisoned_param = minus_params(re_para, -1, adjusted_gradient)
        find_para.append(poisoned_param)
    
    return find_para



