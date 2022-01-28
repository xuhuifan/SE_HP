using Distributions
using Random
using JLD2
using EvalMetrics
using Statistics
using SpecialFunctions
using Plots


function plot_change_time(sending_time_list, t_i_s_list, dataNum)
    sending_num = Array{Int64}(undef, dataNum)
    change_num = Array{Int64}(undef, dataNum)

    for ii = 1:dataNum
        sending_num[ii] = length(sending_time_list[ii])
        change_num[ii] = length(t_i_s_list[ii])
    end
    rank_1 = sortperm(sending_num)
    # plot(sending_num[rank_1], change_num[rank_1], seriestype = :scatter)
    p1 = histogram(sending_num[rank_1], xlabel = "sending number")
    p2 = histogram(change_num[rank_1], xlabel = "change time number")
    plot(p1, p2, layout = (1, 2), legend = false)
    savefig("sending_time_versus_change_time.png") 


end

function calculate_position(tp_i ::Float64, tp_list::Array{Float64, 1})
    # tp_list should be sorted
    index_c = 1
    for time_i in tp_list
        if tp_i>time_i
            index_c += 1
        else
            break
        end
    end

    return index_c

end


function changing_time_related!(t_i_s_list ::Array{Array{Float64,N} where N,1}, send_node ::Array{Int64, 1}, receive_node::Array{Int64, 1}, eventtime::Array{Float64, 1}, EdgeNum::Int64, sender_receiver_num::Array{Int64, 2})
    # sender_receiver_num = zeros(Int64, EdgeNum, 2)
    for tt = 1:EdgeNum
        send_i = calculate_position(eventtime[tt], t_i_s_list[send_node[tt]])
        receive_i = calculate_position(eventtime[tt], t_i_s_list[receive_node[tt]])

        sender_receiver_num[tt, :] = [send_i, receive_i]
    end

    # return sender_receiver_num
end


function calculate_interval(ii::Int64, KK::Int64, dataNum::Int64, t_i_s_list::Array{Array{Float64,N} where N,1}, C_list ::Array{Array{Array{Float64,1},1},1}, b_ij ::Array{Int64,1}, u_ij ::Array{Int64,2}, send_to_i_time ::Array{Array{Float64,N} where N,1}, send_to_i_index ::Array{Array{Int64,N} where N,1}, receive_from_i_time ::Array{Array{Float64,N} where N,1}, receive_from_i_index ::Array{Array{Int64,N} where N,1}, start_time ::Float64, end_time ::Float64)
    integral_val_tt = zeros(Float64, KK)

    for jj = 1:dataNum
        if ii != jj
            index_1 = 1
            index_2 = 1
            for t_i_time in (t_i_s_list[jj])
                if start_time > t_i_time
                    index_1 += 1
                end
                if end_time > t_i_time
                    index_2 += 1
                else
                    break
                end
            end

            if index_1 == index_2
                integral_val_tt .+= C_list[jj][index_1]*(end_time-start_time)
            else
                jj_num = index_1:index_2
                jj_time = t_i_s_list[jj][index_1:(index_2-1)]
                integral_val_tt .+= cal_jj_integral(C_list[jj], jj_time, jj_num, start_time, end_time, KK)
            end
        end
    end

    ll_previous_num = zeros(Float64, KK)

    for time_i_index in 1:length(send_to_i_time[ii])
        time_i = send_to_i_time[ii][time_i_index]
        if (time_i > start_time)&(time_i < end_time)
            jj_val_i = send_to_i_index[ii][time_i_index]
            if (b_ij[jj_val_i]==0)
                for k1 = 1:KK
                    if (u_ij[jj_val_i, 2]==k1)
                        ll_previous_num[k1] += 1.
                    end
                end
            end
        end
        if time_i > end_time
            break
        end
    end

    for time_i_index in 1:length(receive_from_i_time[ii])
        time_i = receive_from_i_time[ii][time_i_index]
        if (time_i > start_time)&(time_i < end_time)
            jj_val_i = receive_from_i_index[ii][time_i_index]
            if (b_ij[jj_val_i]==0)
                for k1 = 1:KK
                    if (u_ij[jj_val_i, 1]==k1)
                        ll_previous_num[k1] += 1.
                    end
                end
            end
        end
        if time_i > end_time
            break
        end
    end

    return integral_val_tt, ll_previous_num

end



function traning_testing_split!(training_ratio, send_node, receive_node, eventtime)
    training_num = Int64.(floor(length(send_node)*training_ratio))

    send_node_test = deepcopy(send_node[(training_num+1):end])
    receive_node_test = deepcopy(receive_node[(training_num+1):end])
    eventtime_test = deepcopy(eventtime[(training_num+1):end])

    send_node = deepcopy(send_node[1:(training_num)])
    receive_node = deepcopy(receive_node[1:(training_num)])
    eventtime = deepcopy(eventtime[1:(training_num)])

    # println("size of send_node is: ", size(send_node))
    # unique_node_id = sort(unique([hcat(reshape(send_node, length(send_node)), reshape(receive_node, length(receive_node))]))
    unique_node_id = sort(unique([send_node;receive_node]))
    # print(unique_node_id)

    dataNum_new = length(unique_node_id)

    for ii = 1:dataNum_new
        for jj = 1:length(send_node)
            if send_node[jj]==unique_node_id[ii]
                send_node[jj] = ii
            end
            if receive_node[jj]==unique_node_id[ii]
                receive_node[jj] = ii
            end
        end
    end

    remove_judge = trues(length(send_node_test))
    for ii = 1:length(send_node_test)
        if (!(send_node_test[ii] in unique_node_id))|(!(receive_node_test[ii] in unique_node_id))
            remove_judge[ii] = false
        end
    end
    send_node_test = send_node_test[remove_judge]
    receive_node_test = receive_node_test[remove_judge]
    eventtime_test = eventtime_test[remove_judge]


    for ii = 1:dataNum_new
        for jj = 1:length(send_node_test)
            if send_node_test[jj]==unique_node_id[ii]
                send_node_test[jj] = ii
            end
            if receive_node_test[jj]==unique_node_id[ii]
                receive_node_test[jj] = ii
            end
        end
    end

    union_connect = []
    for tt = 1:length(send_node)
        append!(union_connect, [[send_node[tt], receive_node[tt]]])
    end
    union_connect = unique(union_connect)

    unique_send_connect = []
    unique_receive_connect = []
    for tt = 1:length(union_connect)
        append!(unique_send_connect, union_connect[tt][1])
        append!(unique_receive_connect, union_connect[tt][2])
    end

    for ii = 1:dataNum_new
        append!(unique_send_connect, ii)
        append!(unique_receive_connect, ii)
    end

    return send_node, receive_node, eventtime, unique_send_connect, unique_receive_connect, send_node_test, receive_node_test, eventtime_test, dataNum_new
end

function CRT(y::Int64, psi::Float64)
    return_val = 0
    for ii = 1:y
        if rand()<(psi/(psi+ii-1))
            return_val += 1
        end
    end
    return return_val
end

struct HDBN
    TT ::Float64
    EdgeNum ::Int64
    dataNum ::Int64
    KK ::Int64
    send_node_position ::Array{Array{Int64,N} where N,1}
    sending_to_nodes_list ::Array{Array{Int64,N} where N,1}
    sending_to_nodes_list_coeff ::Array{Array{Int64,N} where N,1}
    receiving_from_nodes_list ::Array{Array{Int64,N} where N,1}
    receiving_from_nodes_list_coeff ::Array{Array{Int64,N} where N,1}
    sending_time_list ::Array{Array{Float64,N} where N,1}
    mutually_exciting_pair ::Array{Array{Int64,N} where N,1}
    receive_j_len ::Array{Int64,1}
    max_receive_len ::Int64
    mutually_len ::Array{Int64,1}
    max_mutually_len ::Int64
    send_node ::Array{Int64,1}
    receive_node ::Array{Int64,1}
    eventtime ::Array{Float64,1}
    unique_send_connect ::Array{Int64,1}
    unique_receive_connect ::Array{Int64,1}
    send_to_i_time::Array{Array{Float64,N} where N,1}
    send_to_i_index::Array{Array{Int64,N} where N,1}
    receive_from_i_time::Array{Array{Float64,N} where N,1}
    receive_from_i_index::Array{Array{Int64,N} where N,1}
end

function test_process(myarray, NumEdge_pseudo)
    send_node = Array{Int64}(undef, NumEdge_pseudo)
    receive_node = Array{Int64}(undef, NumEdge_pseudo)
    eventtime = Array{Float64}(undef, NumEdge_pseudo)
    judges_tt = trues(NumEdge_pseudo)
    for tt = 1:NumEdge_pseudo
        if myarray[tt,1]!=myarray[tt,2]
            send_node[tt] = Int64(myarray[tt, 1])
            receive_node[tt] = Int64(myarray[tt, 2])
            eventtime[tt] = myarray[tt,3]/(24*3600.0)
        else
            judges_tt[tt] = false
        end
    end
    send_node = send_node[judges_tt]
    receive_node = receive_node[judges_tt]
    eventtime = eventtime[judges_tt]
    eventtime_minimum = minimum(eventtime)
    for tt = 1:length(eventtime)
        eventtime[tt] -= eventtime_minimum
    end
    unique_node = sort(unique(vcat(send_node, receive_node)))

    dataNum = length(unique_node)

    for u_i = 1:length(unique_node)
        judge_index = findall(==(unique_node[u_i]), send_node)
        send_node[judge_index] .= u_i

        judge_index = findall(==(unique_node[u_i]), receive_node)
        receive_node[judge_index] .= u_i
    end

    sort_index = sortperm(eventtime)
    send_node = send_node[sort_index]
    receive_node = receive_node[sort_index]
    eventtime = eventtime[sort_index]

    return send_node, receive_node, eventtime, unique_node, dataNum

end



function model_ini(send_node, receive_node, eventtime, dataNum, EdgeNum, KK, unique_send_connect, unique_receive_connect)
    tau_v = 3.0
    TT = eventtime[end]+0.01
    a_M = 1000.
    b_M = 1.
    M_seq = rand(Gamma(a_M, 1.), dataNum)/b_M

    a_alpha = 1.
    b_alpha = 1.
    alpha_v = rand(Gamma(a_alpha, 1.))/b_alpha

    a_Lambda = [1.]
    b_Lambda = [a_M/b_M*dataNum]

    c₀ = [1.]
    r₀ = [1.]
    rₖ = rand(Gamma(r₀[1]/KK, 1.), KK)./c₀[1]
    η = [1.]
    ξ = [1.]
    Lambdas = zeros(Float64, KK, KK)
    for k1 = 1:KK
        for k2 = 1:KK
            if k1==k2
                Lambdas[k1, k2] = rand(Gamma(ξ[1]*rₖ[k1], 1.))/η[1]
            else
                Lambdas[k1, k2] = rand(Gamma(rₖ[k1]*rₖ[k2], 1.))/η[1]
            end
        end
    end

    a_beta_11 = [1.0]
    b_beta_11 = [1.0]
    a_beta_12 = [1.0]
    b_beta_12 = [1.0]
    betas = zeros(Float64, length(unique_send_connect))
    for tt = 1:length(unique_send_connect)
        if unique_send_connect[tt]==unique_receive_connect[tt]
            betas[tt] = rand(Gamma(a_beta_11[1], 1.))/b_beta_11[1]
        else
            betas[tt] = rand(Gamma(a_beta_12[1], 1.))/b_beta_12[1]
        end
    end

    pis0 = ones(Float64, dataNum, KK)/KK


    sending_to_nodes_list = Array{Int64}[[] for _ in 1:dataNum]
    sending_to_nodes_list_coeff = Array{Int64}[[] for _ in 1:dataNum]
    receiving_from_nodes_list = Array{Int64}[[] for _ in 1:dataNum]
    receiving_from_nodes_list_coeff = Array{Int64}[[] for _ in 1:dataNum]
    for tt = 1:length(unique_send_connect)
        append!(receiving_from_nodes_list[unique_send_connect[tt]], unique_receive_connect[tt])
        append!(receiving_from_nodes_list_coeff[unique_send_connect[tt]], tt)
        append!(sending_to_nodes_list[unique_receive_connect[tt]], unique_send_connect[tt])
        append!(sending_to_nodes_list_coeff[unique_receive_connect[tt]], tt)
    end

    poisson_i_value = rand(Gamma(2, 1.))
    number_i_list = Array{Int64}(undef, dataNum)
    t_i_s_list = Array{Float64}[[] for _ in 1:dataNum]
    t_i_s_nodes = Array{Int64}[[] for _ in 1:dataNum]
    t_i_s_nodes_rank = Array{Int64}[[] for _ in 1:dataNum]
    for ii = 1:dataNum
        number_i_list[ii] = rand(Poisson(poisson_i_value))+1
        t_i_s_list[ii] = sort(rand(number_i_list[ii])).*TT
        t_i_s_nodes[ii] = ones(Int64, number_i_list[ii])*ii
        t_i_s_nodes_rank[ii] = 1:number_i_list[ii]
    end

    flattern_t_i_s_list = collect(Iterators.flatten(t_i_s_list))
    flattern_t_i_s_nodes = collect(Iterators.flatten(t_i_s_nodes))
    flattern_t_i_s_nodes_rank = collect(Iterators.flatten(t_i_s_nodes_rank))

    time_rank = sortperm(flattern_t_i_s_list)
    flattern_t_i_s_list = flattern_t_i_s_list[time_rank]
    flattern_t_i_s_nodes = flattern_t_i_s_nodes[time_rank]
    flattern_t_i_s_nodes_rank = flattern_t_i_s_nodes_rank[time_rank]


    pis = rand(Dirichlet(ones(Float64, KK).*0.1), dataNum)'
    pis_list = [[deepcopy(pis[ii, :])] for ii in 1:dataNum]
    C_list = [[deepcopy(pis[ii, :])] for ii in 1:dataNum]
    for ii = 1:dataNum
        for kk = 1:KK
            C_list[ii][1][kk] = rand(Poisson(M_seq[ii]*pis[ii, kk]))+1.0
        end
    end

    C_ii = Array{Float64}(undef, KK)
    pi_tt = Array{Float64}(undef, KK)
    for tt = 1:length(flattern_t_i_s_nodes)
        sending_nodes_tt = sending_to_nodes_list[flattern_t_i_s_nodes[tt]]
        sending_nodes_tt_rank = sending_to_nodes_list_coeff[flattern_t_i_s_nodes[tt]]
        concentration_parameter = zeros(Float64, KK)
        for ii_index = 1:length(sending_nodes_tt)
            vv = sending_nodes_tt_rank[ii_index]
            concentration_parameter .+= betas[vv].*pis_list[sending_nodes_tt[ii_index]][end]
        end
        for kk = 1:KK
            pi_tt[kk] = rand(Gamma(concentration_parameter[kk]+10^(-10), 1.))+10^(-16)
        end
        append!(pis_list[flattern_t_i_s_nodes[tt]], [deepcopy(pi_tt./sum(pi_tt))])

        for kk = 1:KK
            # println("pis_list[flattern_t_i_s_nodes[tt]][end][kk] is: ", pis_list[flattern_t_i_s_nodes[tt]][end][kk])
            C_ii[kk] = rand(Poisson(M_seq[flattern_t_i_s_nodes[tt]]*pis_list[flattern_t_i_s_nodes[tt]][end][kk]+10^(-10)))+1.0
        end
        if sum(C_ii)==0
            println("wrong!")
        end
        append!(C_list[flattern_t_i_s_nodes[tt]], [deepcopy(C_ii)])
    end

    send_node_position = Array{Int64}[ [] for _ in 1:dataNum]
    for ii = 1:dataNum
        send_node_position[ii] = findall(send_node.==ii)
    end

    sending_time_list = Array{Float64}[ [] for _ in 1:dataNum]
    for ii in 1:dataNum
        (sending_time_list[ii] = eventtime[send_node.==ii])
    end

    b_ij = zeros(Int64, EdgeNum)

    mutually_exciting_pair = Array{Int64}[[] for _ in 1:EdgeNum]
    for tt = 1:EdgeNum

        current_send_node = send_node[1:(tt-1)]
        current_receive_node = receive_node[1:(tt-1)]

        idx1 = findall((send_node[tt].==current_receive_node).&(receive_node[tt].==current_send_node))
        mutually_exciting_pair[tt] = idx1

    end


    receive_j_len = Array{Int64}(undef, dataNum)
    for ii = 1:dataNum
        receive_j_len[ii] = length(sending_to_nodes_list[ii])
    end
    max_receive_len = maximum(receive_j_len)

    mutually_len = Array{Int64}(undef, EdgeNum)
    for tt = 1:EdgeNum
        mutually_len[tt] = length(mutually_exciting_pair[tt])
    end
    max_mutually_len = maximum(mutually_len)

    send_to_i_time = Array{Float64}[[] for _ in 1:dataNum]
    send_to_i_index = Array{Int64}[[] for _ in 1:dataNum]
    receive_from_i_time = Array{Float64}[[] for _ in 1:dataNum]
    receive_from_i_index = Array{Int64}[[] for _ in 1:dataNum]
    for tt = 1:EdgeNum
        append!(send_to_i_time[receive_node[tt]], eventtime[tt])
        append!(send_to_i_index[receive_node[tt]], tt)
        append!(receive_from_i_time[send_node[tt]], eventtime[tt])
        append!(receive_from_i_index[send_node[tt]], tt)
    end

    # @save "colledge_processed_v1.jld2" time_inteval_receive_j_list time_inteval_receive_j_list_2 time_inteval_non_receive_j_list sending_time_list sender_receiver_num mutually_exciting_pair receiving_j_list time_inteval_receive_j_to_i_list time_inteval_receive_j_to_i_list_2 last_time_inteval_non_receive_j_list last_time_inteval_receive_j_list last_time_inteval_receive_j_list_2 last_time_inteval_receive_j_to_i_list last_time_inteval_receive_j_to_i_list_2 receive_j_len max_receive_len mutually_len max_mutually_len connection_index_send remove_list add_list last_time_rank last_add_list last_time_list

    return tau_v, TT, a_M, b_M, M_seq, a_alpha, b_alpha, alpha_v, Lambdas, betas,
            pis0, sending_to_nodes_list, sending_to_nodes_list_coeff, receiving_from_nodes_list, receiving_from_nodes_list_coeff,
            poisson_i_value, t_i_s_list, flattern_t_i_s_nodes, flattern_t_i_s_nodes_rank,
            pis_list, C_list, send_node_position, sending_time_list, b_ij, mutually_exciting_pair, receive_j_len, max_receive_len, mutually_len, max_mutually_len,
            send_to_i_time, send_to_i_index, receive_from_i_time, receive_from_i_index, c₀, r₀, rₖ, η, ξ, a_beta_11, b_beta_11, a_beta_12, b_beta_12
        

end


function back_propagate!(hdbn::HDBN, pis_list ::Array{Array{Array{Float64,1},1},1}, C_list ::Array{Array{Array{Float64,1},1},1}, betas ::Array{Float64,1}, flattern_t_i_s_nodes ::Array{Int64,1}, flattern_t_i_s_nodes_rank ::Array{Int64,1}, t_i_s_list ::Array{Array{Float64,N} where N,1}, a_beta_11::Array{Float64,1}, b_beta_11::Array{Float64,1}, a_beta_12::Array{Float64,1}, b_beta_12::Array{Float64,1})

    m_ik_list = deepcopy(C_list)

    z_ij = zeros(Float64, length(hdbn.unique_send_connect))

    b_betas_hyper = zeros(Float64, length(hdbn.unique_send_connect))

    a = Array{Float64}(undef, hdbn.KK)

    psi_same_layer = zeros(Float64, hdbn.max_receive_len+1, hdbn.KK)
    # rank_total = Array{Int64}[[] for _ in 1:length(flattern_t_i_s_nodes)]
    for tt = length(flattern_t_i_s_nodes):-1:1

        sending_nodes_tt = hdbn.sending_to_nodes_list[flattern_t_i_s_nodes[tt]]
        sending_nodes_tt_rank = hdbn.sending_to_nodes_list_coeff[flattern_t_i_s_nodes[tt]]

        current_change_point = t_i_s_list[flattern_t_i_s_nodes[tt]][flattern_t_i_s_nodes_rank[tt]]

        rank_seq = ones(Int64, length(sending_nodes_tt))

        vv_seq = Array{Int64}(undef, length(sending_nodes_tt))
        for ii_index = 1:length(sending_nodes_tt)
            vv_seq[ii_index] = sending_nodes_tt_rank[ii_index]

            rank_seq[ii_index] = calculate_position(current_change_point, t_i_s_list[sending_nodes_tt[ii_index]])
            psi_same_layer[ii_index, :] = betas[vv_seq[ii_index]].*pis_list[sending_nodes_tt[ii_index]][rank_seq[ii_index]]
        end
        # rank_total[tt] = rank_seq

        psi_sum = reshape(sum(psi_same_layer[1:length(sending_nodes_tt), :], dims = 1), hdbn.KK)
        para_nn = psi_sum .+ m_ik_list[flattern_t_i_s_nodes[tt]][flattern_t_i_s_nodes_rank[tt]+1] .+ 10^(-16)

        for kk = 1:KK
            a[kk] = rand(Gamma(para_nn[kk], 1.))+10^(-16)
        end
        pis_list[flattern_t_i_s_nodes[tt]][flattern_t_i_s_nodes_rank[tt]+1] = (deepcopy(a./sum(a)))

        psi_para_1 = sum(psi_sum) + 10^(-16)
        latent_count_i = sum(m_ik_list[flattern_t_i_s_nodes[tt]][flattern_t_i_s_nodes_rank[tt]+1])+10^(-16)
        gam1 = rand(Gamma(psi_para_1, 1.))+10^(-16)
        gam2 = rand(Gamma(latent_count_i, 1.))+10^(-16)
        q_i_s = gam1/(gam1 + gam2)

        for ii_index = 1:length(sending_nodes_tt)
            b_betas_hyper[vv_seq[ii_index]] += log(q_i_s)
        end


        for kk = 1:hdbn.KK

            current_val = Int64(m_ik_list[flattern_t_i_s_nodes[tt]][flattern_t_i_s_nodes_rank[tt]+1][kk])

            if current_val>0
                y_i_k = CRT(current_val, psi_sum[kk])

                kk_prob = psi_same_layer[1:length(sending_nodes_tt), kk]

                kk_prob = kk_prob./sum(kk_prob)

                z_ik_J = Float64.(rand(Multinomial(y_i_k, kk_prob)))

                for ii_index = 1:length(sending_nodes_tt)
                    m_ik_list[sending_nodes_tt[ii_index]][rank_seq[ii_index]][kk] += z_ik_J[ii_index]
                    z_ij[vv_seq[ii_index]] += z_ik_J[ii_index]
                end


            end
        end

    end

    for ii = 1:hdbn.dataNum
        m_ii = m_ik_list[ii][1]
        para_nn = 10^(-16) .+ m_ii
        for kk = 1:hdbn.KK
            a[kk] = rand(Gamma(para_nn[kk], 1.))+10^(-16)
        end
        pis_list[ii][1] = deepcopy(a./sum(a))
    end

    l_k_11 = 1.0
    l_k_12 = 1.0
    p_k_11 = 1.0
    p_k_12 = 1.0
    b_b_11 = 1.0
    b_b_12 = 1.0
    b_a_11 = 1.0
    b_a_12 = 1.0
    for ij = 1:length(hdbn.unique_send_connect)
        if hdbn.unique_send_connect[ij]==hdbn.unique_receive_connect[ij]
            l_k_11 += CRT(Int64(z_ij[ij]), a_beta_11[1])
            p_k_11 -= log(b_beta_11[1]/(b_beta_11[1]-b_betas_hyper[ij]))
            b_a_11 += a_beta_11[1]
            b_b_11 -= b_betas_hyper[ij]
        else
            l_k_12 += CRT(Int64(z_ij[ij]), a_beta_12[1])
            p_k_12 -= log(b_beta_12[1]/(b_beta_12[1]-b_betas_hyper[ij]))
            b_a_12 += a_beta_12[1]
            b_b_12 -= b_betas_hyper[ij]
        end
    end

    a_beta_11[1] = rand(Gamma(l_k_11, 1.))/(p_k_11)
    a_beta_12[1] = rand(Gamma(l_k_12, 1.))/(p_k_12)
    b_beta_11[1] = rand(Gamma(b_a_11, 1.))/b_b_11
    b_beta_12[1] = rand(Gamma(b_a_12, 1.))/b_b_12


    for ij = 1:length(hdbn.unique_send_connect)
        if hdbn.unique_send_connect[ij]==hdbn.unique_receive_connect[ij]
            betas[ij] = rand(Gamma(a_beta_11[1]+z_ij[ij], 1.))/(b_beta_11[1]-b_betas_hyper[ij])
        else
            betas[ij] = rand(Gamma(a_beta_12[1]+z_ij[ij], 1.))/(b_beta_12[1]-b_betas_hyper[ij])
        end
    end

    # return rank_total

end


function sample_b_u!(hdbn ::HDBN, C_list ::Array{Array{Array{Float64,1},1},1}, Lambdas ::Array{Float64,2}, alpha_v ::Array{Float64,1}, tau_v ::Array{Float64,1}, b_ij ::Array{Int64,1}, u_ij ::Array{Int64,2}, sender_receiver_num ::Array{Int64, 2})

    rate_vector = Array{Float64}(undef, hdbn.max_mutually_len+hdbn.KK^2)
    for tt = 1:hdbn.EdgeNum

        for k1 = 1:hdbn.KK
            for k2 = 1:hdbn.KK
                rate_vector[k1+ (k2-1)*(hdbn.KK)] = C_list[hdbn.send_node[tt]][sender_receiver_num[tt, 1]][k1]*C_list[hdbn.receive_node[tt]][sender_receiver_num[tt, 2]][k2]*Lambdas[k1,k2]
            end
        end
        for jj =1:hdbn.mutually_len[tt]
            rate_vector[jj+hdbn.KK^2] = alpha_v[1]*exp(-tau_v[1]*(hdbn.eventtime[tt]-hdbn.eventtime[hdbn.mutually_exciting_pair[tt][jj]]))
        end
        # if sum(rate_vector[1:(hdbn.KK^2)])==0
        #     println("C_list[hdbn.send_node[tt]] is: ", C_list[hdbn.send_node[tt]])
        #     println("C_list[hdbn.receive_node[tt]] is: ", C_list[hdbn.receive_node[tt]])
        #     println("C_list[hdbn.send_node[tt]][sender_receiver_num[tt, 1]] is: ", C_list[hdbn.send_node[tt]][sender_receiver_num[tt, 1]])
        #     println("C_list[hdbn.receive_node[tt]][sender_receiver_num[tt, 2]] is: ", C_list[hdbn.receive_node[tt]][sender_receiver_num[tt, 2]])
        #     println("Lambdas is: ", Lambdas)
        #     println("length is: ", hdbn.mutually_len[tt])
        # end

        # println("before sum is: ", sum(rate_vector[1:(hdbn.mutually_len[tt]+hdbn.KK^2)]))
        prob = (rate_vector[1:(hdbn.mutually_len[tt]+hdbn.KK^2)].+10^(-16))./(sum(rate_vector[1:(hdbn.mutually_len[tt]+hdbn.KK^2)])+10^(-16))
        # println("sum of prob is: ", sum(prob))
        prob ./= sum(prob)
        index_v = rand(Categorical(prob))
        
        # try
        #     index_v = rand(Categorical(prob))
            
        # catch
        #     println("sum of prob is: ", (prob))
        # end

        if index_v<=(hdbn.KK^2)
            a = fldmod(index_v, hdbn.KK)
            if a[2] == 0
                u_ij[tt, 1] = hdbn.KK
                u_ij[tt, 2] = a[1]
            else
                u_ij[tt, 1] = a[2]
                u_ij[tt, 2] = a[1]+1
            end
            b_ij[tt] = 0
        else
            b_ij[tt] = index_v - (hdbn.KK^2)
        end

    end

end

function cal_area(KK ::Int64, dataNum ::Int64, C_list ::Array{Array{Array{Float64,1},1},1}, flattern_t_i_s_nodes::Array{Int64, 1}, flattern_t_i_s_nodes_rank ::Array{Int64, 1}, t_i_s_list::Array{Array{Float64,N} where N,1})
    C_ii_column = zeros(Float64, KK)
    C_ii_KK = zeros(Float64, KK, KK)
    for ii = 1:dataNum
        C_ii_column .+= C_list[ii][1]
        C_ii_KK .+= reshape(C_list[ii][1], KK, 1)*reshape(C_list[ii][1], 1, KK)
    end
    C_ii_KK .-= reshape(C_list[flattern_t_i_s_nodes[1]][flattern_t_i_s_nodes_rank[1]], KK, 1)*reshape(C_list[flattern_t_i_s_nodes[1]][flattern_t_i_s_nodes_rank[1]], 1, KK)
    C_ii_KK .+= reshape(C_list[flattern_t_i_s_nodes[1]][flattern_t_i_s_nodes_rank[1]+1], KK, 1)*reshape(C_list[flattern_t_i_s_nodes[1]][flattern_t_i_s_nodes_rank[1]+1], 1, KK)
    C_ii_column .-= C_list[flattern_t_i_s_nodes[1]][flattern_t_i_s_nodes_rank[1]]
    C_ii_column .+= C_list[flattern_t_i_s_nodes[1]][flattern_t_i_s_nodes_rank[1]+1]

    vals_copy = (reshape(C_ii_column, KK, 1)*reshape(C_ii_column, 1, KK) .- C_ii_KK)

    areas = vals_copy.*(t_i_s_list[flattern_t_i_s_nodes[2]][flattern_t_i_s_nodes_rank[2]]-t_i_s_list[flattern_t_i_s_nodes[1]][flattern_t_i_s_nodes_rank[1]])

    C_ii_column .-= (C_list[flattern_t_i_s_nodes[2]][flattern_t_i_s_nodes_rank[2]])
    vals_copy .-= reshape(C_ii_column, KK, 1)* reshape(C_list[flattern_t_i_s_nodes[2]][flattern_t_i_s_nodes_rank[2]], 1, KK)
    vals_copy .-= reshape(C_list[flattern_t_i_s_nodes[2]][flattern_t_i_s_nodes_rank[2]], KK, 1)*reshape(C_ii_column, 1, KK)

    for tt = 2:(length(flattern_t_i_s_nodes)-1)
        vals_copy .+= reshape(C_ii_column, KK, 1)* reshape(C_list[flattern_t_i_s_nodes[tt]][flattern_t_i_s_nodes_rank[tt]+1], 1, KK)
        vals_copy .+= reshape(C_list[flattern_t_i_s_nodes[tt]][flattern_t_i_s_nodes_rank[tt]+1], KK, 1)*reshape(C_ii_column, 1, KK)
        areas .+= vals_copy.*(t_i_s_list[flattern_t_i_s_nodes[tt+1]][flattern_t_i_s_nodes_rank[tt+1]]-t_i_s_list[flattern_t_i_s_nodes[tt]][flattern_t_i_s_nodes_rank[tt]])
        C_ii_column .+= (C_list[flattern_t_i_s_nodes[tt]][flattern_t_i_s_nodes_rank[tt]+1])
        C_ii_column .-= (C_list[flattern_t_i_s_nodes[tt+1]][flattern_t_i_s_nodes_rank[tt+1]])
        vals_copy .-= reshape(C_ii_column, KK, 1)* reshape(C_list[flattern_t_i_s_nodes[tt+1]][flattern_t_i_s_nodes_rank[tt+1]], 1, KK)
        vals_copy .-= reshape(C_list[flattern_t_i_s_nodes[tt+1]][flattern_t_i_s_nodes_rank[tt+1]], KK, 1)*reshape(C_ii_column, 1, KK)
    end

    return areas
end

function sample_Lambda!(hdbn ::HDBN, C_list ::Array{Array{Array{Float64,1},1},1}, b_ij ::Array{Int64,1}, u_ij ::Array{Int64,2}, Lambdas ::Array{Float64,2}, flattern_t_i_s_nodes ::Array{Int64,1}, flattern_t_i_s_nodes_rank ::Array{Int64,1}, t_i_s_list ::Array{Array{Float64,N} where N,1}, c₀ ::Array{Float64, 1}, r₀ ::Array{Float64, 1}, rₖ ::Array{Float64, 1}, η ::Array{Float64, 1}, ξ ::Array{Float64, 1})
    # areas_re = zeros(Float64, hdbn.KK, hdbn.KK)
    # C_ii = zeros(Float64, hdbn.dataNum, hdbn.KK)
    # for ii = 1:hdbn.dataNum
    #     C_ii[ii, :] = C_list[ii][1]
    # end

    # for tt = 1:(length(flattern_t_i_s_nodes)-1)
    #     C_ii[flattern_t_i_s_nodes[tt], :] = C_list[flattern_t_i_s_nodes[tt]][flattern_t_i_s_nodes_rank[tt]+1]
    #     areas_re .+= ((reshape(sum(C_ii, dims = 1), hdbn.KK, 1)*reshape(sum(C_ii, dims = 1), 1, hdbn.KK) .- C_ii'*C_ii).*(t_i_s_list[flattern_t_i_s_nodes[tt+1]][flattern_t_i_s_nodes_rank[tt+1]]-t_i_s_list[flattern_t_i_s_nodes[tt]][flattern_t_i_s_nodes_rank[tt]]))
    # end

    areas = cal_area(hdbn.KK, hdbn.dataNum, C_list, flattern_t_i_s_nodes, flattern_t_i_s_nodes_rank, t_i_s_list)

    # println("area difference is: ", sum(broadcast(abs, areas-areas_re)))

    # println("areas: ", areas)
    # println("b_Lambda is: ", hdbn.b_Lambda)
    # print("type of areas is: ", typeof(areas))

    judge_b_0_u_ij = u_ij[b_ij.==0, :]

    Xₖ = zeros(Int64, hdbn.KK, hdbn.KK)
    lₖ = zeros(Float64, hdbn.KK, hdbn.KK)

    for k1 = 1:hdbn.KK
        for k2 = 1:hdbn.KK
            Xₖ[k1, k2] = sum((judge_b_0_u_ij[:,1].==k1).& (judge_b_0_u_ij[:,2].==k2))
            if k1 == k2
                lₖ[k1, k2] = CRT(Xₖ[k1, k2], ξ[1]*rₖ[k1])
            else
                lₖ[k1, k2] = CRT(Xₖ[k1, k2], rₖ[k1]*rₖ[k2])
            end
        end
    end

    a_eta = 1.
    b_eta = 1.
    for k1 = 1:hdbn.KK
        for k2 = 1:hdbn.KK
            if k1 == k2
                a_eta += ξ[1]*rₖ[k1]
            else
                a_eta += rₖ[k1]*rₖ[k2]
            end
            b_eta += Lambdas[k1,k2]
        end
    end
    η[1] = rand(Gamma(a_eta, 1.))/b_eta

    c₀[1] = rand(Gamma(1.0+r₀[1], 1.))/(1.0+sum(rₖ))

    r_star = r₀[1] + 0.1*randn()
    if r_star >0
        proposal_ll = r_star*(log(c₀[1])-1.0)-hdbn.KK*loggamma(r_star/hdbn.KK)
        current_ll = r₀[1]*(log(c₀[1])-1.0)-hdbn.KK*loggamma(r₀[1]/hdbn.KK)
        for k1 = 1:hdbn.KK
            proposal_ll += r_star/hdbn.KK*log(rₖ[k1])
            current_ll += r₀[1]/hdbn.KK*log(rₖ[k1])
        end
        if log(rand())<(proposal_ll-current_ll)
            r₀[1] = r_star
        end
    end

    a_xi = 1.0
    b_xi = 1.0
    for k1 = 1:hdbn.KK
        a_xi += lₖ[k1, k1]
        b_xi -= rₖ[k1]*log(η[1]/(η[1]+areas[k1,k1]))
    end
    ξ[1] = rand(Gamma(a_xi, 1.))/b_xi

    for k1 = 1:hdbn.KK
        a_r = r₀[1]/hdbn.KK + sum(lₖ[k1])
        b_r = deepcopy(c₀[1])
        for k2 = 1:hdbn.KK
            if k1 == k2
                b_r -= ξ[1]*log(η[1]/(η[1]+areas[k1,k2]))
            else
                b_r -= rₖ[k2]*log(η[1]/(η[1]+areas[k1,k2]))
            end
        end
        rₖ[k1] = rand(Gamma(a_r, 1.))/b_r
    end


    for k1 = 1:hdbn.KK
        for k2 = 1:hdbn.KK
            # try
            if k1 == k2
                Lambdas[k1,k2] = rand(Gamma(ξ[1]*rₖ[k1]+Xₖ[k1, k2]+10^(-16), 1.))/(η[1]+areas[k1,k2])
            else
                Lambdas[k1,k2] = rand(Gamma(rₖ[k1]*rₖ[k2]+Xₖ[k1, k2]+10^(-16), 1.))/(η[1]+areas[k1,k2])
            end
                    
            # catch
            #     println("ξ[1] is: ", ξ[1])
            #     println("η[1] is: ", η[1])
            #     println("rₖ[k1] is: ", rₖ[k1])
            #     println("rₖ[k2] is: ", rₖ[k2])
            # end
        end
    end


end

# function sample_Lambda_hyper!(Lambdas::Array{Float64,2}, a_Lambda::Array{Float64,1}, b_Lambda::Array{Float64,1}, M_seq::Array{Float64, 1}, KK::Int64)
#     b_Lambda[1] = rand(InverseGamma((KK^2)*a_Lambda[1] + (mean(M_seq)^2), sum(Lambdas)+1.))
#     # we use random-walk Metropolis-Hastings to re-sample a_Lambda[1]
#     new_val = a_Lambda[1] + randn()*0.05
#     log_sum_Lambda = 0.
#     for k1 = 1:KK
#         for k2 = 1:KK
#             log_sum_Lambda += log(Lambdas[k1, k2])
#         end
#     end
#     if new_val > 0
#         ll_current = a_Lambda[1]*log_sum_Lambda - (KK^2)*(loggamma(a_Lambda[1])+a_Lambda[1]*log(b_Lambda[1]))
#         ll_proposal = new_val*log_sum_Lambda - (KK^2)*(loggamma(new_val)+new_val*log(b_Lambda[1]))

#         if log(rand())<(ll_proposal-ll_current+logpdf(Gamma(0.1, 0.1), new_val)-logpdf(Gamma(0.1, 0.1), a_Lambda[1]))
#             a_Lambda[1] = new_val
#         end
    
#     end


# end

function cal_jj_integral(C_list_jj::Array{Array{Float64,1},1}, jj_time::Array{Float64, 1}, jj_num::UnitRange{Int64}, s_time::Float64, current_time::Float64, KK::Int64)
    if length(jj_time)==1
        integral_1 = C_list_jj[jj_num[1]].*(jj_time[1]-s_time)
        integral_2 = C_list_jj[jj_num[2]].*(current_time - jj_time[1])
        integral_val = integral_1 .+ integral_2
    else
        integral_1 = C_list_jj[jj_num[1]].*(jj_time[1]-s_time)
        integral_2 = C_list_jj[jj_num[end]].*(current_time - jj_time[end])
        integral_3 = zeros(Float64, KK)
        for ij in 1:(length(jj_time)-1)
            integral_3 .+= C_list_jj[jj_num[ij+1]].*(jj_time[ij+1]-jj_time[ij])
        end
        integral_val = integral_1 .+ integral_2 .+ integral_3
    end
    return integral_val
end

function touchard_sample(a1::Float64, a2::Float64, prior::Float64, log_proportions::Array{Float64, 1}, proportions::Array{Float64, 1})
    if a2 ==0
        select_val = rand(Poisson(prior*exp(a1)))
    else
        log_v = 0
        for ci = 1:length(log_proportions)
            log_v += log(ci)
            log_proportions[ci] = ci*(a1+log(prior))+a2*log(ci)-log_v
        end
        # println("a1 is: ", a1)
        # println("a2 is: ", a2)
        # println("prior is: ", prior)
        # println("maximu of log proportions is: ", maximum(log_proportions))

        log_proportions .-= maximum(log_proportions)
        for log_index = 1:length(log_proportions)
            proportions[log_index] = exp(log_proportions[log_index])
        end
        # println("Proportions is: ", proportions)
        select_val = rand(Categorical(proportions./sum(proportions)))
    end

    return Float64(select_val)
end

function sample_C_list!(hdbn ::HDBN, C_list ::Array{Array{Array{Float64,1},1},1}, pis_list ::Array{Array{Array{Float64,1},1},1}, M_seq ::Array{Float64, 1}, b_ij ::Array{Int64,1}, u_ij ::Array{Int64,2}, Lambdas ::Array{Float64,2}, flattern_t_i_s_nodes ::Array{Int64,1}, flattern_t_i_s_nodes_rank ::Array{Int64,1}, t_i_s_list ::Array{Array{Float64,N} where N,1})

    CanNum = 1000
    log_proportions = Array{Float64}(undef, CanNum)
    proportions = Array{Float64}(undef, CanNum)

    for tt = 1:length(flattern_t_i_s_nodes)
        current_time = t_i_s_list[flattern_t_i_s_nodes[tt]][flattern_t_i_s_nodes_rank[tt]]
        if flattern_t_i_s_nodes_rank[tt] == 1
            s_time = 0.
        else
            s_time = t_i_s_list[flattern_t_i_s_nodes[tt]][flattern_t_i_s_nodes_rank[tt]-1]
        end

        integral_val_tt, ll_previous_num = calculate_interval(flattern_t_i_s_nodes[tt], hdbn.KK, hdbn.dataNum, t_i_s_list, C_list, b_ij, u_ij, hdbn.send_to_i_time, hdbn.send_to_i_index, hdbn.receive_from_i_time, hdbn.receive_from_i_index, s_time, current_time)

        for k1 = 1:hdbn.KK
            prior = M_seq[flattern_t_i_s_nodes[tt]]*pis_list[flattern_t_i_s_nodes[tt]][flattern_t_i_s_nodes_rank[tt]][k1]
            a1 = -sum(integral_val_tt.*(Lambdas[:, k1]+Lambdas[k1, :]))
            a2 = ll_previous_num[k1]

            C_list[flattern_t_i_s_nodes[tt]][flattern_t_i_s_nodes_rank[tt]][k1] = touchard_sample(a1, a2, prior, log_proportions, proportions)
        end

    end


    for jj = 1:hdbn.dataNum
        if length(t_i_s_list[jj])==0
            s_time = 0.
        else
            s_time = t_i_s_list[jj][end]
        end
        integral_val_tt, ll_previous_num = calculate_interval(jj, hdbn.KK, hdbn.dataNum, t_i_s_list, C_list, b_ij, u_ij, hdbn.send_to_i_time, hdbn.send_to_i_index, hdbn.receive_from_i_time, hdbn.receive_from_i_index, s_time, hdbn.TT)

        for k1 = 1:hdbn.KK
            prior = M_seq[jj]*pis_list[jj][end][k1]
            a1 = -sum(integral_val_tt.*(Lambdas[:, k1]+Lambdas[k1, :]))
            a2 = ll_previous_num[k1]

            C_list[jj][end][k1] = touchard_sample(a1, a2, prior, log_proportions, proportions)
        end


    end

end


function sample_M_seq!(hdbn ::HDBN, C_list ::Array{Array{Array{Float64,1},1},1}, M_seq ::Array{Float64, 1}, a_H ::Array{Float64, 1}, b_H ::Array{Float64, 1})
    lens_seq = zeros(Float64, hdbn.dataNum)
    counts_seq = zeros(Float64, hdbn.dataNum)
    l_H = 0.
    p_k_H = 0.
    for ii = 1:hdbn.dataNum
        lens_seq[ii] = length(C_list[ii])

        for C_i_s in C_list[ii]
            counts_seq[ii] += sum(C_i_s)
        end
        l_H += CRT(Int64(counts_seq[ii]), a_H[1])
        p_k_H -= log(b_H[1]/(b_H[1]+lens_seq[ii]))
    end

    a_H[1] = rand(Gamma(500.0 + l_H, 1.))/(1.0 + p_k_H)
    b_H[1] = rand(Gamma(1.0 + hdbn.dataNum*a_H[1], 1.))/(1.0 + sum(M_seq))

    for ii = 1:hdbn.dataNum
        M_seq[ii] = rand(Gamma(counts_seq[ii] + a_H[1], 1.))/(b_H[1]+lens_seq[ii])
    end

end

function sample_alpha!(hdbn ::HDBN, b_ij ::Array{Int64,1}, tau_v ::Array{Float64,1}, alpha_v ::Array{Float64,1}, a_alpha ::Float64, b_alpha::Float64)
    h_tau = 0
    b_ij_over_1_sum = 0
    for tt = 1:hdbn.EdgeNum
        h_tau += (1-exp(-tau_v[1]*(hdbn.TT-hdbn.eventtime[tt])))
        if (b_ij[tt]>0)
            b_ij_over_1_sum += 1
        end
    end
    h_tau *= tau_v[1]^(-1)

    alpha_v[1] = rand(Gamma(a_alpha+b_ij_over_1_sum, 1.))/(b_alpha+h_tau)

    # return alpha_v

end


function sample_tau!(hdbn ::HDBN, tau_v ::Array{Float64,1}, b_ij ::Array{Int64,1}, alpha_v ::Array{Float64,1})
    a_tau = 0.1
    b_tau = 0.1

    new_tau_v = tau_v[1] + 0.1*randn()
    if new_tau_v>0
        ll_old = 0
        ll_new = 0

        for tt = 1:hdbn.EdgeNum
            if b_ij[tt]>0
                ll_old -= tau_v[1]*(hdbn.eventtime[tt]-hdbn.eventtime[hdbn.mutually_exciting_pair[tt][b_ij[tt]]])
                ll_new -= new_tau_v*(hdbn.eventtime[tt]-hdbn.eventtime[hdbn.mutually_exciting_pair[tt][b_ij[tt]]])

            end
            ll_old -= alpha_v[1]/tau_v[1]*(1-exp(-tau_v[1]*(hdbn.TT-hdbn.eventtime[tt])))
            ll_new -= alpha_v[1]/new_tau_v*(1-exp(-new_tau_v*(hdbn.TT-hdbn.eventtime[tt])))
        end

        ll_old += logpdf(Gamma(a_tau, 1/b_tau), tau_v[1])
        ll_new += logpdf(Gamma(a_tau, 1/b_tau), new_tau_v)

        if log(rand())<(ll_new-ll_old)
            tau_v[1] = new_tau_v
        end
    end

    # return tau_v
end


function sample_poisson_i_val!(t_i_s_list ::Array{Array{Float64,N} where N,1}, poisson_i_value ::Array{Float64, 1}, dataNum::Int64)
    len_sum = 0.
    for t_i_i in t_i_s_list
        len_sum += length(t_i_i)
    end

    poisson_i_value[1] = rand(Gamma(0.1+len_sum, 1.))/(dataNum+0.1)
end


function cal_pi_j_given_pi_i(ii::Int64, rank_i::Int64, concentration_parameter::Array{Array{Float64,1},1}, diff_ii::Array{Array{Float64,1},1}, sending_nodes_tt::Array{Int64,1}, sending_node_tt_coeff::Array{Int64,1}, receive_i::Int64, change_time_i::Array{Float64, 1}, t_i_s_list::Array{Array{Float64,N} where N,1}, betas::Array{Float64,1}, pis_list::Array{Array{Array{Float64,1},1},1}, input_pi::Array{Float64, 1})

    for ii_index = 1:length(sending_nodes_tt)
        vv = sending_node_tt_coeff[ii_index]

        rank_ii = calculate_position(change_time_i[rank_i], t_i_s_list[sending_nodes_tt[ii_index]])

        concentration_parameter[1] .+= betas[vv].*pis_list[sending_nodes_tt[ii_index]][rank_ii]
        if sending_nodes_tt[ii_index] == ii
            diff_ii[1] = betas[vv].*(input_pi.-pis_list[sending_nodes_tt[ii_index]][rank_ii])
        end
    end

    delta_current_ll = logpdf(Dirichlet(concentration_parameter[1]), pis_list[receive_i][rank_i+1])
    delta_proposal_ll = logpdf(Dirichlet(concentration_parameter[1].+diff_ii[1].+10^(-16)), pis_list[receive_i][rank_i+1])

    return delta_current_ll, delta_proposal_ll
end


function sample_K!(hdbn ::HDBN, C_list ::Array{Array{Array{Float64,1},1},1}, pis_list ::Array{Array{Array{Float64,1},1},1}, M_seq ::Array{Float64, 1}, b_ij ::Array{Int64,1}, u_ij ::Array{Int64,2}, Lambdas ::Array{Float64,2}, t_i_s_list ::Array{Array{Float64,N} where N,1}, poisson_i_value ::Array{Float64, 1})
    for ii = 1:hdbn.dataNum
        if rand()<0.5 # choose to propose a new change change point

            new_time = rand()*hdbn.TT
            current_index = calculate_position(new_time, t_i_s_list[ii])

            concentration_parameter = zeros(Float64, KK)

            sending_nodes_tt = sending_to_nodes_list[ii]
            for ii_index = 1:length(sending_nodes_tt)
                vv = sending_to_nodes_list_coeff[ii][ii_index]
                rank_ii = calculate_position(new_time, t_i_s_list[sending_nodes_tt[ii_index]])
                concentration_parameter .+= betas[vv].*pis_list[sending_nodes_tt[ii_index]][rank_ii]
            end

            C_ii = Array{Float64}(undef, KK)
            pi_tt = Array{Float64}(undef, KK)
            for kk = 1:KK
                pi_tt[kk] = rand(Gamma(concentration_parameter[kk]+10^(-10), 1.))+10^(-16)
            end
            pi_tt = pi_tt./sum(pi_tt)
            for kk = 1:KK
                C_ii[kk] = rand(Poisson(M_seq[ii]*pi_tt[kk]+10^(-10)))
            end

            if current_index >length(t_i_s_list[ii])
                next_time = hdbn.TT
            else
                next_time = t_i_s_list[ii][current_index]
            end

            integral_val_tt, ll_previous_num = calculate_interval(ii, hdbn.KK, hdbn.dataNum, t_i_s_list, C_list, b_ij, u_ij, hdbn.send_to_i_time, hdbn.send_to_i_index, hdbn.receive_from_i_time, hdbn.receive_from_i_index, new_time, next_time)

            current_ll = 0.
            proposal_ll = 0.

            # the probability of P(πⱼ|πᵢ)
            receive_nodes_tt = receiving_from_nodes_list[ii]
            diff_ii = [zeros(Float64, hdbn.KK)]
            concentration_parameter = [zeros(Float64, KK)]

            for receive_i in receive_nodes_tt
                change_time_i = t_i_s_list[receive_i]
                start_rank = calculate_position(new_time, change_time_i)
                end_rank = calculate_position(next_time, change_time_i)

                sending_nodes_tt = sending_to_nodes_list[receive_i]
                sending_nodes_tt_coeff = sending_to_nodes_list_coeff[receive_i]

                for rank_i = (start_rank:(end_rank-1))

                    concentration_parameter[1] = zeros(Float64, KK)
                    diff_ii[1] = zeros(Float64, hdbn.KK)

                    delta_current_ll, delta_proposal_ll = cal_pi_j_given_pi_i(ii, rank_i, concentration_parameter, diff_ii, sending_nodes_tt, sending_nodes_tt_coeff, receive_i, change_time_i, t_i_s_list, betas, pis_list, pi_tt)

                    current_ll += delta_current_ll
                    proposal_ll += delta_proposal_ll

                end

            end


            for k1 = 1:hdbn.KK
                for k2 = 1:hdbn.KK
                    current_ll -= C_list[ii][current_index][k1]*integral_val_tt[k2]*(Lambdas[k1,k2]+Lambdas[k2,k1])
                    proposal_ll -= C_ii[k1]*integral_val_tt[k2]*(Lambdas[k1,k2]+Lambdas[k2,k1])
                end
            end
            for k1 = 1:hdbn.KK
                current_ll += ll_previous_num[k1]*log(C_list[ii][current_index][k1])
                proposal_ll += ll_previous_num[k1]*log(C_ii[k1])
            end

            if log(rand())<(proposal_ll - current_ll + log(poisson_i_value[1])-log(length(t_i_s_list[ii])+1))
                # update pis_list, C_list, t_i_s_list
                splice!(pis_list[ii], current_index:(current_index-1), [pi_tt])
                splice!(C_list[ii], current_index:(current_index-1), [C_ii])
                splice!(t_i_s_list[ii], current_index:(current_index-1), new_time)
                # println("node ii add one change point: ", ii)
            end

        else # choose to delete an existing change point
            if length(t_i_s_list[ii])>0
                current_index = rand(Categorical(ones(length(t_i_s_list[ii]))./length(t_i_s_list[ii])))
                new_time = t_i_s_list[ii][current_index]

                if current_index ==length(t_i_s_list[ii])
                    next_time = hdbn.TT
                else
                    next_time = t_i_s_list[ii][current_index+1]
                end

                integral_val_tt, ll_previous_num = calculate_interval(ii, hdbn.KK, hdbn.dataNum, t_i_s_list, C_list, b_ij, u_ij, hdbn.send_to_i_time, hdbn.send_to_i_index, hdbn.receive_from_i_time, hdbn.receive_from_i_index, new_time, next_time)

                current_ll = 0.
                proposal_ll = 0.


                # the probability of P(πⱼ|πᵢ)
                receive_nodes_tt = receiving_from_nodes_list[ii]
                diff_ii = [zeros(Float64, hdbn.KK)]

                concentration_parameter = [zeros(Float64, KK)]

                input_pi = pis_list[ii][current_index]
                for receive_i in receive_nodes_tt
                    change_time_i = t_i_s_list[receive_i]
                    end_rank = calculate_position(next_time, change_time_i)
                    start_rank = calculate_position(new_time, change_time_i)

                    sending_nodes_tt = sending_to_nodes_list[receive_i]
                    sending_nodes_tt_coeff = sending_to_nodes_list_coeff[receive_i]

                    if receive_i != ii
                        for rank_i = (start_rank:(end_rank-1))
                            concentration_parameter[1] = zeros(Float64, KK)
                            diff_ii[1] = zeros(Float64, hdbn.KK)

                            delta_current_ll, delta_proposal_ll = cal_pi_j_given_pi_i(ii, rank_i, concentration_parameter, diff_ii, sending_nodes_tt, sending_nodes_tt_coeff, receive_i, change_time_i, t_i_s_list, betas, pis_list, input_pi)

                            current_ll += delta_current_ll
                            proposal_ll += delta_proposal_ll

                        end
                    else
                        rank_i  = end_rank
                        if (rank_i < length(change_time_i))
                            concentration_parameter[1] = zeros(Float64, KK)
                            diff_ii[1] = zeros(Float64, hdbn.KK)

                            delta_current_ll, delta_proposal_ll = cal_pi_j_given_pi_i(ii, rank_i, concentration_parameter, diff_ii, sending_nodes_tt, sending_nodes_tt_coeff, receive_i, change_time_i, t_i_s_list, betas, pis_list, input_pi)

                            current_ll += delta_current_ll
                            proposal_ll += delta_proposal_ll

                        end


                    end
                end


                for k1 = 1:hdbn.KK
                    for k2 = 1:hdbn.KK
                        current_ll -= C_list[ii][current_index+1][k1]*integral_val_tt[k2]*(Lambdas[k1,k2]+Lambdas[k2,k1])
                        proposal_ll -= C_list[ii][current_index][k1]*integral_val_tt[k2]*(Lambdas[k1,k2]+Lambdas[k2,k1])
                    end
                end
                for k1 = 1:hdbn.KK
                    current_ll += ll_previous_num[k1]*log(C_list[ii][current_index+1][k1])
                    proposal_ll += ll_previous_num[k1]*log(C_list[ii][current_index][k1])
                end

                if log(rand())<(proposal_ll - current_ll - log(poisson_i_value[1])+log(length(t_i_s_list[ii])))
                    # update pis_list, C_list, t_i_s_list
                    deleteat!(pis_list[ii], current_index)
                    deleteat!(C_list[ii], current_index)
                    deleteat!(t_i_s_list[ii], current_index)

                    # println("node ii remove one change point: ", ii)
                end

            end
        end
    end

end

function sample_change_point!(hdbn ::HDBN, C_list ::Array{Array{Array{Float64,1},1},1}, pis_list ::Array{Array{Array{Float64,1},1},1}, b_ij ::Array{Int64,1}, u_ij ::Array{Int64,2}, Lambdas ::Array{Float64,2}, t_i_s_list ::Array{Array{Float64,N} where N,1})
    for ii = 1:hdbn.dataNum
        for jj = 1:length(t_i_s_list[ii])
            propose_time_point = t_i_s_list[ii][jj] + randn()*0.1*hdbn.TT
            if (propose_time_point>0)&(propose_time_point<hdbn.TT)
                current_index = calculate_position(propose_time_point, t_i_s_list[ii])

                if current_index == (jj) # t_i_s_list[ii][jj-1] < propose_time_point < t_i_s_list[ii][jj]

                    current_ll = 0.
                    proposal_ll = 0.

########################

                    # the probability of P(πⱼ|πᵢ)
                    receive_nodes_tt = receiving_from_nodes_list[ii]
                    diff_ii = [zeros(Float64, hdbn.KK)]

                    concentration_parameter = [zeros(Float64, KK)]

                    input_pi = pis_list[ii][current_index+1]
                    for receive_i in receive_nodes_tt
                        change_time_i = t_i_s_list[receive_i]
                        end_rank = calculate_position(t_i_s_list[ii][jj], change_time_i)
                        start_rank = calculate_position(propose_time_point, change_time_i)

                        sending_nodes_tt = sending_to_nodes_list[receive_i]
                        sending_nodes_tt_coeff = sending_to_nodes_list_coeff[receive_i]

                        for rank_i = (start_rank:(end_rank-1))
        
                            concentration_parameter[1] = zeros(Float64, KK)
                            diff_ii[1] = zeros(Float64, hdbn.KK)
        
                            delta_current_ll, delta_proposal_ll = cal_pi_j_given_pi_i(ii, rank_i, concentration_parameter, diff_ii, sending_nodes_tt, sending_nodes_tt_coeff, receive_i, change_time_i, t_i_s_list, betas, pis_list, input_pi)
        
                            current_ll += delta_current_ll
                            proposal_ll += delta_proposal_ll
        

                        end

                    end


########################

                    diff_integral, diff_num = calculate_interval(ii, hdbn.KK, hdbn.dataNum, t_i_s_list, C_list, b_ij, u_ij, hdbn.send_to_i_time, hdbn.send_to_i_index, hdbn.receive_from_i_time, hdbn.receive_from_i_index, propose_time_point, t_i_s_list[ii][jj])
                    for k1 = 1:hdbn.KK
                        for k2 = 1:hdbn.KK
                            current_ll -= C_list[ii][jj][k1]*(diff_integral[k2])*(Lambdas[k1,k2]+Lambdas[k2,k1])
                            proposal_ll -= C_list[ii][jj+1][k1]*(diff_integral[k2])*(Lambdas[k1,k2]+Lambdas[k2,k1])

                        end
                    end
                    for k1 = 1:hdbn.KK
                        current_ll += (diff_num[k1])*log(C_list[ii][jj][k1])
                        proposal_ll += (diff_num[k1])*log(C_list[ii][jj+1][k1])
                    end

                    if log(rand())<(proposal_ll - current_ll)
                        # update pis_list, C_list, t_i_s_list
                        # println("t_i_s_list[ii] is: ", t_i_s_list[ii])
                        # println("jj is: ", jj)
                        splice!(t_i_s_list[ii], jj, propose_time_point)
                        # println("t_i_s_list[ii] is: ", t_i_s_list[ii])
                    end

                elseif current_index ==(jj+1) # t_i_s_list[ii][jj] < propose_time_point < t_i_s_list[ii][jj+1]

                    diff_integral, diff_num = calculate_interval(ii, hdbn.KK, hdbn.dataNum, t_i_s_list, C_list, b_ij, u_ij, hdbn.send_to_i_time, hdbn.send_to_i_index, hdbn.receive_from_i_time, hdbn.receive_from_i_index, t_i_s_list[ii][jj], propose_time_point)

                    current_ll = 0.
                    proposal_ll = 0.

########################


                    # the probability of P(πⱼ|πᵢ)
                    receive_nodes_tt = receiving_from_nodes_list[ii]
                    diff_ii = [zeros(Float64, hdbn.KK)]

                    concentration_parameter = [zeros(Float64, KK)]

                    input_pi = pis_list[ii][current_index-1]
                    for receive_i in receive_nodes_tt
                        change_time_i = t_i_s_list[receive_i]
                        end_rank = calculate_position(propose_time_point, change_time_i)
                        start_rank = calculate_position(t_i_s_list[ii][jj], change_time_i)

                        sending_nodes_tt = sending_to_nodes_list[receive_i]
                        sending_nodes_tt_coeff = sending_to_nodes_list_coeff[receive_i]

                        for rank_i = (start_rank:(end_rank-1))
        
                            concentration_parameter[1] = zeros(Float64, KK)
                            diff_ii[1] = zeros(Float64, hdbn.KK)
        
                            delta_current_ll, delta_proposal_ll = cal_pi_j_given_pi_i(ii, rank_i, concentration_parameter, diff_ii, sending_nodes_tt, sending_nodes_tt_coeff, receive_i, change_time_i, t_i_s_list, betas, pis_list, input_pi)
        
                            current_ll += delta_current_ll
                            proposal_ll += delta_proposal_ll
        

                        end

                    end


########################


                    for k1 = 1:hdbn.KK
                        for k2 = 1:hdbn.KK
                            current_ll -= C_list[ii][jj+1][k1]*(diff_integral[k2])*(Lambdas[k1,k2]+Lambdas[k2,k1])
                            proposal_ll -= C_list[ii][jj][k1]*(diff_integral[k2])*(Lambdas[k1,k2]+Lambdas[k2,k1])

                        end
                    end
                    for k1 = 1:hdbn.KK
                        current_ll += (diff_num[k1])*log(C_list[ii][jj+1][k1])
                        proposal_ll += (diff_num[k1])*log(C_list[ii][jj][k1])
                    end

                    if log(rand())<(proposal_ll - current_ll)
                        # println("t_i_s_list[ii] is: ", t_i_s_list[ii])
                        # println("jj is: ", jj)
                        splice!(t_i_s_list[ii], jj, propose_time_point)
                        # println("Hooray")
                        # println("t_i_s_list[ii] is: ", t_i_s_list[ii])
                    end

                # else

                #     concentration_parameter = zeros(Float64, KK)

                #     sending_nodes_tt = sending_to_nodes_list[ii]
                #     sending_nodes_tt_rank = sending_to_nodes_list_coeff[ii]
                #     for ii_index = 1:length(sending_nodes_tt)
                #         vv = sending_nodes_tt_rank[ii_index]
                #         rank_ii = calculate_position(propose_time_point, t_i_s_list[sending_nodes_tt[ii_index]])
                #         concentration_parameter .+= betas[vv].*pis_list[sending_nodes_tt[ii_index]][rank_ii]
                #     end

                #     C_ii = Array{Float64}(undef, KK)
                #     pi_tt = Array{Float64}(undef, KK)
                #     for kk = 1:KK
                #         pi_tt[kk] = rand(Gamma(concentration_parameter[kk]+10^(-10), 1.))+10^(-16)
                #     end
                #     pi_tt = pi_tt./sum(pi_tt)
                #     for kk = 1:KK
                #         # println("pis_list[flattern_t_i_s_nodes[tt]][end][kk] is: ", pis_list[flattern_t_i_s_nodes[tt]][end][kk])
                #         C_ii[kk] = rand(Poisson(M*pi_tt[kk]+10^(-10)))
                #     end

                #     if current_index >length(t_i_s_list[ii])
                #         propose_next_time = hdbn.TT
                #     else
                #         propose_next_time = t_i_s_list[ii][current_index]
                #     end

                #     if jj ==length(t_i_s_list[ii])
                #         current_next_time = hdbn.TT
                #     else
                #         current_next_time = t_i_s_list[ii][jj+1]
                #     end

                #     propose_integral, propose_num = calculate_interval(ii, hdbn.KK, hdbn.dataNum, t_i_s_list, C_list, b_ij, u_ij, hdbn.send_to_i_time, hdbn.send_to_i_index, hdbn.receive_from_i_time, hdbn.receive_from_i_index, propose_time_point, propose_next_time)

                #     current_integral, current_num = calculate_interval(ii, hdbn.KK, hdbn.dataNum, t_i_s_list, C_list, b_ij, u_ij, hdbn.send_to_i_time, hdbn.send_to_i_index, hdbn.receive_from_i_time, hdbn.receive_from_i_index, t_i_s_list[ii][jj], current_next_time)

                #     current_ll = 0.
                #     proposal_ll = 0.
                #     for k1 = 1:hdbn.KK
                #         for k2 = 1:hdbn.KK
                #             current_ll -= C_list[ii][current_index][k1]*propose_integral[k2]*(Lambdas[k1,k2]+Lambdas[k2,k1])
                #             current_ll -= C_list[ii][jj+1][k1]*current_integral[k2]*(Lambdas[k1,k2]+Lambdas[k2,k1])

                #             proposal_ll -= C_ii[k1]*propose_integral[k2]*(Lambdas[k1,k2]+Lambdas[k2,k1])
                #             proposal_ll -= C_list[ii][jj][k1]*current_integral[k2]*(Lambdas[k1,k2]+Lambdas[k2,k1])

                #         end
                #     end
                #     for k1 = 1:hdbn.KK
                #         current_ll += propose_num[k1]*log(C_list[ii][current_index][k1])
                #         current_ll += current_num[k1]*log(C_list[ii][jj+1][k1])
                #         proposal_ll += propose_num[k1]*log(C_ii[k1])
                #         proposal_ll += current_num[k1]*log(C_list[ii][jj][k1])
                #     end

                #     if log(rand())<(proposal_ll - current_ll)
                #         # update pis_list, C_list, t_i_s_list
                #         splice!(pis_list[ii], current_index:(current_index-1), [pi_tt])
                #         splice!(C_list[ii], current_index:(current_index-1), [C_ii])
                #         splice!(t_i_s_list[ii], current_index:(current_index-1), propose_time_point)
                #         if current_index < jj
                #                 deleteat!(pis_list[ii], jj+1)
                #                 deleteat!(C_list[ii], jj+1)
                #                 deleteat!(t_i_s_list[ii], jj+1)
                #         else
                #             deleteat!(pis_list[ii], jj)
                #             deleteat!(C_list[ii], jj)
                #             deleteat!(t_i_s_list[ii], jj)
                #         end
                #     end

                end

            end
        end
    end
end


function reformat_change_points!(t_i_s_list ::Array{Array{Float64,N} where N,1}, flattern_t_i_s_nodes ::Array{Int64,1}, flattern_t_i_s_nodes_rank ::Array{Int64,1})
    t_i_s_nodes = Array{Int64}[[] for _ in 1:dataNum]
    t_i_s_nodes_rank = Array{Int64}[[] for _ in 1:dataNum]
    for ii = 1:dataNum
        t_i_s_nodes[ii] = ones(Int64, length(t_i_s_list[ii]))*ii
        t_i_s_nodes_rank[ii] = 1:length(t_i_s_list[ii])
    end

    flattern_t_i_s_list = collect(Iterators.flatten(t_i_s_list))
    flattern_t_i_s_nodes_re = collect(Iterators.flatten(t_i_s_nodes))
    flattern_t_i_s_nodes_rank_re = collect(Iterators.flatten(t_i_s_nodes_rank))

    time_rank = sortperm(flattern_t_i_s_list)
    if length(time_rank)>length(flattern_t_i_s_nodes)
        append!(flattern_t_i_s_nodes, zeros(length(time_rank)-length(flattern_t_i_s_nodes)))
        append!(flattern_t_i_s_nodes_rank, zeros(length(time_rank)-length(flattern_t_i_s_nodes_rank)))
    else
        deleteat!(flattern_t_i_s_nodes, (length(time_rank)+1):length(flattern_t_i_s_nodes))
        deleteat!(flattern_t_i_s_nodes_rank, (length(time_rank)+1):length(flattern_t_i_s_nodes_rank))
    end
    # flattern_t_i_s_list = flattern_t_i_s_list[time_rank]
    for tt = 1:length(time_rank)
        flattern_t_i_s_nodes[tt] = flattern_t_i_s_nodes_re[time_rank[tt]]
        flattern_t_i_s_nodes_rank[tt] = flattern_t_i_s_nodes_rank_re[time_rank[tt]]
    end

end

function ll_cal!(hdbn ::HDBN, b_ij ::Array{Int64,1}, alpha_v ::Array{Float64,1}, tau_v ::Array{Float64,1}, u_ij ::Array{Int64,2}, Lambdas ::Array{Float64,2}, C_list ::Array{Array{Array{Float64,1},1},1}, sender_receiver_num::Array{Int64, 2}, flattern_t_i_s_nodes ::Array{Int64,1}, flattern_t_i_s_nodes_rank ::Array{Int64,1}, t_i_s_list ::Array{Array{Float64,N} where N,1})
    ll_val_part_1 = 0
    ll_val_part_2 = 0
    ll_val_part_3 = 0
    ll_val_part_4 = 0

    for tt = 1:hdbn.EdgeNum
        if b_ij[tt]==0
            # try
            ll_val_part_1 += log(C_list[hdbn.send_node[tt]][sender_receiver_num[tt, 1]][u_ij[tt, 1]]*
            C_list[hdbn.receive_node[tt]][sender_receiver_num[tt, 2]][u_ij[tt, 2]]*Lambdas[u_ij[tt, 1], u_ij[tt, 2]])

            # catch
            #     println("tt is: ", tt)
            #     println("ii is: ", hdbn.send_node[tt])
            #     println("C_list[hdbn.send_node[tt]] is: ", C_list[hdbn.send_node[tt]])
            #     println("sender_receiver_num[tt, 1] is: ", sender_receiver_num[tt, 1])
            #     println("----------------------------------------")
            #     println("")
            # end
        else
            ll_val_part_2 -= tau_v[1]*(hdbn.eventtime[tt]-hdbn.eventtime[mutually_exciting_pair[tt][b_ij[tt]]])
            ll_val_part_2 += log(alpha_v[1])
        end
    end

    areas = cal_area(hdbn.KK, hdbn.dataNum, C_list, flattern_t_i_s_nodes, flattern_t_i_s_nodes_rank, t_i_s_list)

    ll_val_part_3 -= sum(areas.*Lambdas)

    h_tau = 0
    for tt = 1:hdbn.EdgeNum
        h_tau += (1-exp(-tau_v[1]*(hdbn.TT-hdbn.eventtime[tt])))
    end
    h_tau *= tau_v[1]^(-1)

    ll_val_part_4 -= alpha_v[1]*h_tau

    return ll_val_part_1, ll_val_part_2, ll_val_part_3, ll_val_part_4

end


function calculate_AUC(test_time_range, dataNum, KK, send_node_test, receive_node_test, eventtime_test, send_node, receive_node, eventtime, C_list_end, Lambdas, alpha_v, tau_v)

    historical_time_pair = [[[] for _ in 1:dataNum] for _ in 1:dataNum]
    for tt = 1:EdgeNum
        append!(historical_time_pair[send_node[tt]][receive_node[tt]], [eventtime[tt]])
    end
    repeat_time = 100
    precision_seq = []
    auc_seq = []
    for rei in 1:repeat_time
        val = sort(rand(2))
        lower_v = test_time_range[1] + val[1]*(test_time_range[2]-test_time_range[1])
        higher_v = test_time_range[1] + val[2]*(test_time_range[2]-test_time_range[1])

        rr_index = findall((eventtime_test.>lower_v).&(eventtime_test.<higher_v))
        send_node_rr = send_node_test[rr_index]
        receive_node_rr = receive_node_test[rr_index]
        judge_index = ones(Int64, dataNum, dataNum)
        for ii = 1:dataNum
            judge_index[ii, ii] = 0
        end
        true_val = zeros(Float64, dataNum, dataNum)
        # println("dataNum is: ", dataNum)
        for rr_index = 1:length(send_node_rr)
            true_val[send_node_rr[rr_index], receive_node_rr[rr_index]] = 1.
        end
        if sum(true_val.==1)>0
            rate_val = zeros(Float64,dataNum, dataNum)
            for ii = 1:dataNum
                for jj = 1:dataNum
                    if ii!= jj
                        rate = 0.
                        for k1 = 1:KK
                            for k2 = 1:KK
                                rate += C_list_end[ii, k1]*Lambdas[k1,k2]*C_list_end[jj, k2]
                            end
                        end
                        rate *= (higher_v-lower_v)
                        for ij_time in historical_time_pair[jj][ii]
                            rate += alpha_v[1]/tau_v[1]*(exp(-tau_v[1]*(lower_v-ij_time))-exp(-tau_v[1]*(higher_v-ij_time)))
                        end
                        rate_val[ii, jj] = rate
                    end
                end
            end

            true_val_seq = true_val[judge_index.==1]
            predict_val_seq = rate_val[judge_index.==1]

            analysis_result = binary_eval_report(true_val_seq, predict_val_seq)
            append!(precision_seq, analysis_result["precision@fpr0.05"])
            append!(auc_seq, analysis_result["au_roccurve"])
        end

    end

    return mean(precision_seq), mean(auc_seq)

end
