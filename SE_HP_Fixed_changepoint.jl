using DelimitedFiles
using Glob
using JLD2
import Random
import JSON
using Plots

# Random.seed!(3)
include("utilities_v6.jl")
include("generative_process.jl")

Simulation_Flag = false
data_i = 5

if Simulation_Flag
    dataNum = 20
    T = 0.5
    alpha_origin = 1.0
    tau_origin = 3.
    KK = 2
    M = 1000

    send_node, receive_node, eventtime, Lambdas_origin, alpha_origin, tau_origin, betas_origin, dataNum, unique_send_connect, 
        unique_receive_connect, CC_seq, pis_seq = Formal_generative_process(dataNum, T, alpha_origin, tau_origin, KK, M )
    # @load "simulation.jld2"
else
    pathss = "../Hawkes_DirPPs/"
    filess = glob("*.txt", pathss)

    println(filess)

    # myarray=(open(readdlm,filess[1]))
    start_time = time()
    myarray = readdlm(filess[data_i])
    println("Elapsed time of data reading is: ", time() - start_time)
    println(size(myarray))
    println(myarray[1:3])
    NumEdge_pseudo = 10000
    send_node, receive_node, eventtime, unique_node, dataNum = test_process(myarray, NumEdge_pseudo)

end

training_ratio = 0.5


send_node, receive_node, eventtime, unique_send_connect, unique_receive_connect, send_node_test, receive_node_test, eventtime_test, dataNum = 
    traning_testing_split!(training_ratio, send_node, receive_node, eventtime)
    
KK = 10

EdgeNum = length(eventtime)

println("EdgeNum is: ", EdgeNum)
println("dataNum is: ", dataNum )


start_time = time()

tau_v, TT, a_M, b_M, M_seq, a_alpha, b_alpha, alpha_v, Lambdas, betas, 
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

IterationTime = 2000

u_ij = zeros(Int64, hdbn.EdgeNum, 2)
alpha_v = [alpha_v]
tau_v = [tau_v]
start_time = [start_time]
# M = 1000.
# pis_list, C_list, b_ij, betas, gammas, Lambdas, M, alpha_v, tau_v

ll_seq_1 = Array{Float64}(undef, IterationTime)
ll_seq_2 = Array{Float64}(undef, IterationTime)
ll_seq_3 = Array{Float64}(undef, IterationTime)
ll_seq_4 = Array{Float64}(undef, IterationTime)

start_time[1] = time()

burnInTime = Int64(floor(IterationTime/2))
mean_Lambdas = [zeros(Float64, hdbn.KK, hdbn.KK)]
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



visTime = 100
time_stops = hdbn.TT ./ visTime .* (1:visTime)

pis_time_val = zeros(Float64, visTime, dataNum, KK)

a_H =[500.]
b_H = [1.]
running_time = zeros(Float64, IterationTime)
for ite = 1:IterationTime
    # println("Iteration ", ite, " start.")
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
    sample_b_u!(hdbn, C_list, Lambdas, alpha_v, tau_v, b_ij, u_ij, sender_receiver_num)
    # println("Sample_b_u finished.")

    back_propagate!(hdbn, pis_list, C_list, betas, flattern_t_i_s_nodes, flattern_t_i_s_nodes_rank, t_i_s_list, a_beta_11, b_beta_11, a_beta_12, b_beta_12)
    # println("back_propagate finished.")

    sample_Lambda!(hdbn, C_list, b_ij, u_ij, Lambdas, flattern_t_i_s_nodes, flattern_t_i_s_nodes_rank, t_i_s_list, c₀, r₀, rₖ, η, ξ)
    # println("sample_Lambda finished.")

    sample_C_list!(hdbn, C_list, pis_list, M_seq, b_ij, u_ij, Lambdas, flattern_t_i_s_nodes, flattern_t_i_s_nodes_rank, t_i_s_list)
    # println("sample_C_list finished.")

    sample_M_seq!(hdbn, C_list, M_seq, a_H, b_H)
    # println("sample_C_list finished.")

    sample_alpha!(hdbn, b_ij, tau_v, alpha_v, a_alpha, b_alpha)
    # println("sample_alpha finished.")

    sample_tau!(hdbn, tau_v, b_ij, alpha_v)
    # println("sample_tau finished.")

    sample_K!(hdbn, C_list, pis_list, M_seq, b_ij, u_ij, Lambdas, t_i_s_list, poisson_i_value)
    # println("sample_K finished.")

    sample_poisson_i_val!(t_i_s_list, poisson_i_value, hdbn.dataNum)

    sample_change_point!(hdbn, C_list, pis_list, b_ij, u_ij, Lambdas, t_i_s_list)
    # println("sample_change_point finished.")

    reformat_change_points!(t_i_s_list, flattern_t_i_s_nodes, flattern_t_i_s_nodes_rank)
    # println("reformat_change_points finished.")

    changing_time_related!(t_i_s_list, send_node, receive_node, eventtime, EdgeNum, sender_receiver_num)
    # println("reformat_change_points finished.")
    running_time[ite] = time()-ss

    if ite > burnInTime
        mean_Lambdas[1] = (mean_Lambdas[1].*(ite - burnInTime-1).+Lambdas)./(ite-burnInTime)
        mean_M_seq[1] = (mean_M_seq[1].*(ite - burnInTime-1).+M_seq)./(ite-burnInTime)
        mean_alpha[1] = (mean_alpha[1].*(ite - burnInTime-1)+alpha_v[1])/(ite-burnInTime)
        mean_taus[1] = (mean_taus[1].*(ite - burnInTime-1)+tau_v[1])/(ite-burnInTime)
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

        for time_i_index in 1:visTime
            for ii = 1:dataNum
                ii_position = calculate_position(time_stops[time_i_index], t_i_s_list[ii])
                for kk = 1:KK
                    # println("pis_time_val[time_i_index][ii][kk] is: ", pis_time_val[time_i_index,ii,kk])
                    # println("pis_list[ii][ii_position][kk] is: ", pis_list[ii][ii_position][kk])
                    # println("(pis_time_val[time_i_index][ii][kk]*(ite - burnInTime-1)+pis_list[ii][ii_position][kk]) is: ", (pis_time_val[time_i_index][ii][kk]*(ite - burnInTime-1)+pis_list[ii][ii_position][kk]))
                    # println("(pis_time_val[time_i_index][ii][kk]*(ite - burnInTime-1)+pis_list[ii][ii_position][kk])/(ite-burnInTime) is: ", (pis_time_val[time_i_index][ii][kk]*(ite - burnInTime-1)+pis_list[ii][ii_position][kk])/(ite-burnInTime))

                    pis_time_val[time_i_index,ii,kk] = (pis_time_val[time_i_index,ii,kk]*(ite - burnInTime-1)+pis_list[ii][ii_position][kk])/(ite-burnInTime)
                end
            end
        end

    end

    ll_val_part_1, ll_val_part_2, ll_val_part_3, ll_val_part_4 = ll_cal!(hdbn, b_ij, alpha_v, tau_v, u_ij, Lambdas, C_list, sender_receiver_num, flattern_t_i_s_nodes, flattern_t_i_s_nodes_rank, t_i_s_list)
    ll_seq_1[ite] = ll_val_part_1
    ll_seq_2[ite] = ll_val_part_2
    ll_seq_3[ite] = ll_val_part_3
    ll_seq_4[ite] = ll_val_part_4
end
ll_seq = ll_seq_1 .+ ll_seq_2 .+ ll_seq_3 .+ ll_seq_4

dict1 = Dict("sending_num" => sending_num, "receiving_num" => receiving_num, "mean_change_num" => mean_change_num, "pis_time_val" => pis_time_val, "Lambdas" => mean_Lambdas[1], "M_seq" => mean_M_seq[1], "mean_poisson_i_value"=>mean_poisson_i_value[1], "running_time"=>running_time, "ll_seq" => ll_seq)
stringdata = JSON.json(dict1)

open(join(["total_result_", string(data_i), ".json"]), "w") do f
        write(f, stringdata)
end            

# plot_change_time(sending_time_list, t_i_s_list, hdbn.dataNum)

println("alpha value is: ", alpha_v[1])
println("tau value is: ", tau_v[1])

# plot(1:IterationTime, ll_seq_1)
# savefig("myplot_1.png") 

# plot(1:IterationTime, ll_seq_2)
# savefig("myplot_2.png") 

# plot(1:IterationTime, ll_seq_3)
# savefig("myplot_3.png") 

# plot(1:IterationTime, ll_seq_4)
# savefig("myplot_4.png") 

plot(1:IterationTime, ll_seq)
savefig(join(["full_plot_", string(data_i), ".pdf"])) 

test_time_range = [eventtime_test[1], eventtime_test[end]]
results = calculate_AUC(test_time_range, dataNum, KK, send_node_test, receive_node_test, eventtime_test, send_node, receive_node, eventtime, mean_C_list[1], mean_Lambdas[1], mean_alpha[1], mean_taus[1])    

println("AUC is: ", results[2])
println("Precision is: ", results[1])

