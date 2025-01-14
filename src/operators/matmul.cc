#include "operators/matmul.h"

namespace infini
{

    MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA,
                         bool transB)
        : OperatorObj(OpType::MatMul, TensorVec{A, B}, {C}),
          transA(transA), transB(transB)
    {
        IT_ASSERT(checkValid(graph));
    }

    string MatmulObj::toString() const
    {
        std::ostringstream os;
        os << "Matmul([" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B]")
           << ",A=" << inputs[0]->getGuid()
           << ",B=" << inputs[1]->getGuid() << ",C=" << outputs[0]->getGuid()
           << ",mnk=[" << m << "," << n << "," << k << "])";
        return os.str();
    }

    optional<vector<Shape>> MatmulObj::inferShape(const TensorVec &inputs)
    {
        // =================================== 作业 ===================================
        // TODO：返回经过 matmul 操作后的 shape
        // REF: https://github.com/onnx/onnx/blob/main/docs/Operators.md#gemm
        // =================================== 作业 ===================================
        auto A = inputs[0];
        auto B = inputs[1];
        auto A_dims = A->getDims();
        auto B_dims = B->getDims();
        auto A_rank = A->getRank();
        // auto B_rank = B->getRank();
        auto output_dims = A_dims;
        for(size_t i = 0; i < A_rank - 2; i++)
            output_dims[i] = std::max(A_dims[i], B_dims[i]);
        if (transA){
            output_dims[A_rank - 2] = A_dims[A_rank - 1];
        }else{
            output_dims[A_rank - 2] = A_dims[A_rank - 2];
        }

        if (transB){
            output_dims[A_rank - 1] = B_dims[A_rank - 2];
        }else{
            output_dims[A_rank - 1] = B_dims[A_rank - 1];
        }

        return vector<Shape>{output_dims};
    }

} // namespace infini