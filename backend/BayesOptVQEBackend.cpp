#include <boost/functional/hash.hpp>
#include <unordered_map>
#include <memory>
#include "MPIProvider.hpp"
#include "BayesOptVQEBackend.hpp"
#include <boost/math/constants/constants.hpp>

namespace xacc {
namespace vqe {

const VQETaskResult BayesOptVQEBackend::minimize(Eigen::VectorXd parameters) {

	VQETaskResult r;
	const int dim = parameters.size();
	auto pi = boost::math::constants::pi<double>();
	auto computeTask = std::make_shared<ComputeEnergyVQETask>(program);

	bayesopt::Parameters par;
	par = initialize_parameters_to_default();
	par.n_iterations = 190;
	par.random_seed = 0;
  
	VQEBayesOptFunction f(par, computeTask, dim);
  
	// Map parameters to boost vector
	
	boost::numeric::ublas::vector<double> result(dim);
	double* p = parameters.data();
	std::copy(result.begin(), result.end(), p);
  
	boost::numeric::ublas::vector<double> lowerBound(dim);
	boost::numeric::ublas::vector<double> upperBound(dim);
	for (int i = 0; i < dim; i++) {lowerBound[i] = -1*pi;upperBound[i] = pi;}

	f.setBoundingBox(lowerBound,upperBound);
	f.optimize(result);

	xacc::info("BayesOpt VQE Backend finds E = " + std::to_string(0.0));
	return r;

}

}
}
