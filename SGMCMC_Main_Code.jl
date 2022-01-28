using DelimitedFiles
using Glob
using JLD2
import Random
import JSON
using Plots
using Distributions

# Random.seed!(3)
include("utilities_SGMCMC.jl")
include("generative_process.jl")

data_i = 4
Name_seq = ["college", "email", "ubuntu", "overflow"]

pathss = "../Hawkes_DirPPs/"
filess = glob("*.txt", pathss)

println(filess)

# myarray=(open(readdlm,filess[1]))
start_time = time()
myarray = readdlm(filess[data_i])
println("Elapsed time of data reading is: ", time() - start_time)
println(size(myarray))
println(myarray[1:3])
NumEdge_pseudo = 20000
send_node, receive_node, eventtime, unique_node, dataNum = test_process(myarray, NumEdge_pseudo)

IterationTime = 1000

training_ratio = 0.5


KK = 10

send_node, receive_node, eventtime, unique_send_connect, unique_receive_connect, send_node_test, receive_node_test, eventtime_test, dataNum = 
    traning_testing_split!(training_ratio, send_node, receive_node, eventtime)
    

EdgeNum = length(eventtime)

println("EdgeNum is: ", EdgeNum)
println("dataNum is: ", dataNum )

start_time = time()

log_τ, TT, a_M, b_M, M_seq, a_alpha, b_alpha, log_α, log_Λ, log_β, 
            pis0, sending_to_nodes_list, sending_to_nodes_list_coeff, receiving_from_nodes_list, receiving_from_nodes_list_coeff, 
            poisson_i_value, t_i_s_list, flattern_t_i_s_nodes, flattern_t_i_s_nodes_rank, 
            pis_list, C_list, send_node_position, sending_time_list, b_ij, mutually_exciting_pair, receive_j_len, max_receive_len, 
            mutually_len, max_mutually_len, send_to_i_time, send_to_i_index, receive_from_i_time, receive_from_i_index, c₀, r₀, rₖ, η, ξ, a_beta_11, b_beta_11, a_beta_12, b_beta_12 = 
            model_ini(send_node, receive_node, eventtime, dataNum, EdgeNum, KK, unique_send_connect, unique_receive_connect)

println("Elapsed time of model initialization is: ", time() - start_time)
# println("type of poisson_i_value is: ", typeof(poisson_i_value))
# println("type of flattern_t_i_s_nodes is: ", typeof(flattern_t_i_s_nodes))
# println("type of flattern_t_i_s_nodes_rank is: ", typeof(flattern_t_i_s_nodes_rank))

const hdbn = HDBN(TT, EdgeNum, dataNum, KK, send_node_position, sending_to_nodes_list, sending_to_nodes_list_coeff, receiving_from_nodes_list, receiving_from_nodes_list_coeff, 
                  sending_time_list, mutually_exciting_pair, receive_j_len, max_receive_len, mutually_len, max_mutually_len, send_node, receive_node, eventtime, 
                  unique_send_connect, unique_receive_connect, send_to_i_time, send_to_i_index, receive_from_i_time, receive_from_i_index)


poisson_i_value = [poisson_i_value]
sender_receiver_num = Array{Int64}(undef, EdgeNum, 2)
# println("type of t_i_s_list is: ", typeof(t_i_s_list))
# println("type of send_node is: ", typeof(send_node))
# println("type of eventtime is: ", typeof(eventtime))
# println("type of sender_receiver_num is: ", typeof(sender_receiver_num))

changing_time_related!(t_i_s_list, send_node, receive_node, eventtime, EdgeNum, sender_receiver_num)
# println("type of sender_receiver_num is: ", typeof(sender_receiver_num))

fixed_exogeneous = false 

u_ij = zeros(Int64, hdbn.EdgeNum, 2)
log_α = [log_α]
log_τ = [log_τ]
start_time = [start_time]
# M = 1000.
# pis_list, C_list, b_ij, log_β, gammas, log_Λ, M, log_α, log_τ

ll_seq_1 = Array{Float64}(undef, IterationTime)
ll_seq_2 = Array{Float64}(undef, IterationTime)
ll_seq_3 = Array{Float64}(undef, IterationTime)
ll_seq_4 = Array{Float64}(undef, IterationTime)

start_time[1] = time()

burnInTime = Int64(floor(IterationTime-200))
mean_log_Λ = [zeros(Float64, hdbn.KK, hdbn.KK)]
mean_alpha = [0.]
mean_taus = [0.]
mean_C_list = [zeros(Float64, hdbn.dataNum, hdbn.KK)]
mean_M_seq = [zeros(Float64, dataNum)]
mean_poisson_i_value = [0.]


sending_num = Array{Int64}(undef, dataNum)
for ii = 1:dataNum
    sending_num[ii] = length(sending_time_list[ii])
end
receiving_num = Array{Float64}(undef, dataNum)
for ii in 1:dataNum
    receiving_num[ii] = sum(receive_node.==ii)
end

change_num = Array{Int64}(undef, dataNum)
mean_change_num = zeros(Float64, dataNum)



visTime = 500
time_stops = hdbn.TT ./ visTime .* (1:visTime)

pis_time_val = zeros(Float64, visTime, dataNum, KK)

a_H =[100.]
b_H = [1.]
running_time = zeros(Float64, IterationTime)

subset_ratio = 0.02
p_Λ = randn(hdbn.KK, hdbn.KK)
p_β = randn(length(log_β))
p_α = [rand()]
p_τ = [rand()]
V = Float64(hdbn.KK^2 + length(log_β) +2)
A = 1.
ϕ = [1.]
h = 0.01

for ite = 1:IterationTime
    # println("Iteration ", ite, " start.")
    # Lambdas .+= (10.)^(-1)
    if mod(ite, 10)==0
        println("a_H is: ", a_H[1])
        println("b_H is: ", b_H[1])
        # println("Elapsed time of one iteration of Gibbs sampling is: ", time() - start_time[1])
        println("Iteration ", ite, " start.")
        println("-----------------------------------------------")
        println()
        # start_time[1] = time()
    end
    ss = time()

    start_points = rand(Uniform())*hdbn.TT*(1-subset_ratio)
    end_points = start_points + subset_ratio*hdbn.TT
    subset_index = findall((eventtime.>start_points).&(eventtime.<end_points))
    
    sample_b_u!(hdbn, C_list, log_Λ, log_α, log_τ, b_ij, u_ij, sender_receiver_num, a_H, subset_index)
    # println("Sample_b_u finished.")

    back_propagate!(hdbn, pis_list, C_list, log_β, flattern_t_i_s_nodes, flattern_t_i_s_nodes_rank, t_i_s_list, a_beta_11, b_beta_11, a_beta_12, b_beta_12, start_points, end_points)
    # println("back_propagate finished.")

    sample_C_list!(hdbn, C_list, pis_list, M_seq, b_ij, u_ij, log_Λ, flattern_t_i_s_nodes, flattern_t_i_s_nodes_rank, t_i_s_list, a_H, start_points, end_points)
    # println("sample_C_list finished.")

    noise_h = randn()*sqrt(h)
    update_Lambda!(hdbn, C_list, b_ij, u_ij, log_Λ, flattern_t_i_s_nodes, flattern_t_i_s_nodes_rank, t_i_s_list, c₀, r₀, rₖ, η, ξ, a_H, start_points, end_points, subset_index, p_Λ, noise_h, A, h, ϕ)
    # println("sample_Lambda finished.")
    update_beta!(hdbn, pis_list, log_β, flattern_t_i_s_nodes, flattern_t_i_s_nodes_rank, t_i_s_list, start_points, end_points, p_β, noise_h, A, h, ϕ)

    update_α_τ!(hdbn, b_ij, log_τ, log_α, start_points, end_points, subset_index, p_α, p_τ, noise_h, A, h, ϕ)

    Δp = sum(p_Λ.^2)+sum(p_β.^2)+p_α[1]^2+p_τ[1]^2

    ϕ[1] = ϕ[1] + (Δp/V -1)*h

    # sample_M_seq!(hdbn, C_list, log_Λ, M_seq, a_H, b_H)
    # println("sample_M_seq finished.")

    # sample_alpha!(hdbn, b_ij, log_τ, log_α, a_alpha, b_alpha)
    # # println("sample_alpha finished.")

    # sample_tau!(hdbn, log_τ, b_ij, log_α)
    # # println("sample_tau finished.")

    sample_K!(hdbn, C_list, pis_list, M_seq, b_ij, u_ij, log_Λ, t_i_s_list, poisson_i_value, a_H)
    # println("sample_K finished.")

    sample_poisson_i_val!(t_i_s_list, poisson_i_value, hdbn.dataNum)

    sample_change_point!(hdbn, C_list, pis_list, b_ij, u_ij, log_Λ, t_i_s_list, a_H, start_points, end_points)
    # println("sample_change_point finished.")

    reformat_change_points!(t_i_s_list, flattern_t_i_s_nodes, flattern_t_i_s_nodes_rank)
    # println("reformat_change_points finished.")

    changing_time_related!(t_i_s_list, send_node, receive_node, eventtime, EdgeNum, sender_receiver_num)
    # println("reformat_change_points finished.")
    running_time[ite] = time()-ss

    if ite > burnInTime
        mean_Lambdas[1] = (mean_Lambdas[1].*(ite - burnInTime-1).+exp.(log_Λ))./(ite-burnInTime)
        mean_M_seq[1] = (mean_M_seq[1].*(ite - burnInTime-1).+M_seq)./(ite-burnInTime)
        mean_alpha[1] = (mean_alpha[1].*(ite - burnInTime-1)+exp(log_α[1]))/(ite-burnInTime)
        mean_taus[1] = (mean_taus[1].*(ite - burnInTime-1)+exp(log_τ[1]))/(ite-burnInTime)
        mean_poisson_i_value[1] = (mean_poisson_i_value[1].*(ite - burnInTime-1)+poisson_i_value[1])/(ite-burnInTime)
        for i1 = 1:hdbn.dataNum
            mean_C_list[1][i1, :] = (mean_C_list[1][i1, :].*(ite - burnInTime-1).+C_list[i1][end])./(ite-burnInTime)
        end

        for ii = 1:dataNum
            change_num[ii] = length(t_i_s_list[ii])
        end
    
        for ii = 1:dataNum
            mean_change_num[ii] = (mean_change_num[ii]*(ite - burnInTime-1)+change_num[ii])/(ite-burnInTime)
        end


    end

    # ll_val_part_1, ll_val_part_2, ll_val_part_3, ll_val_part_4 = ll_cal!(hdbn, b_ij, log_α, log_τ, u_ij, log_Λ, C_list, sender_receiver_num, flattern_t_i_s_nodes, flattern_t_i_s_nodes_rank, t_i_s_list, a_H)
    # ll_seq_1[ite] = ll_val_part_1
    # ll_seq_2[ite] = ll_val_part_2
    # ll_seq_3[ite] = ll_val_part_3
    # ll_seq_4[ite] = ll_val_part_4
end


test_time_range = [eventtime_test[1], eventtime_test[end]]
results = calculate_AUC(test_time_range, dataNum, KK, send_node_test, receive_node_test, eventtime_test, send_node, receive_node, eventtime, mean_C_list[1], mean_Lambdas[1], mean_alpha[1], mean_taus[1], a_H)    
println("AUC is: ", results[2])
auc_val = results[2]

dict1 = Dict("mean_alpha" => mean_alpha, "mean_taus" => mean_taus,"sending_num" => sending_num,"receiving_num" => receiving_num, "mean_change_num" => mean_change_num, "pis_time_val" => pis_time_val, "Lambdas" => mean_Lambdas[1], "M_seq" => mean_M_seq[1], "mean_poisson_i_value"=>mean_poisson_i_value[1], "running_time"=>running_time, "ll_seq" => ll_seq, "AUC" => auc_val)
stringdata = JSON.json(dict1)

open(join(["new_total_result_", Name_seq[data_i], ".json"]), "w") do f
        write(f, stringdata)
end            

# plot_change_time(sending_time_list, t_i_s_list, hdbn.dataNum)

println("alpha value is: ", exp(log_α[1]))
println("tau value is: ", exp(log_τ[1]))

# plot(1:IterationTime, ll_seq_1)
# savefig("myplot_1.png") 

# plot(1:IterationTime, ll_seq_2)
# savefig("myplot_2.png") 

# plot(1:IterationTime, ll_seq_3)
# savefig("myplot_3.png") 

# plot(1:IterationTime, ll_seq_4)
# savefig("myplot_4.png") 




