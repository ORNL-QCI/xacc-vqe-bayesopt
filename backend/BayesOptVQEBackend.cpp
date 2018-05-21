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
	par.n_iterations = xacc::optionExists("bo-n-iter") ? std::stoi(xacc::getOption("bo-n-iter")) : par.n_iterations;
	par.noise = xacc::optionExists("bo-noise") ? std::stod(xacc::getOption("bo-noise")) : par.noise;
	par.l_type = xacc::optionExists("bo-learn-type") ? str2learn(xacc::getOption("bo-learn-type").c_str()) : par.l_type;
	par.n_init_samples = xacc::optionExists("bo-n-init-iter") ? std::stoi(xacc::getOption("bo-n-init-iter")) : par.n_init_samples;
	par.verbose_level = xacc::optionExists("bo-verbose-level") ? std::stoi(xacc::getOption("bo-verbose-level")) : par.verbose_level;
	par.epsilon = xacc::optionExists("bo-epsilon") ? std::stod(xacc::getOption("bo-epsilon")) : par.epsilon;
  
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

	r.energy = f.getValueAtMinimum();

	auto resultAngles = f.getFinalResult();
	const double * data = &(resultAngles.data()[0]);
        r.angles = Eigen::Map<const Eigen::VectorXd>(data, parameters.size());
	
	std::cout << "RESULT: " << r.angles << ", " << r.energy << "\n";
//	xacc::info("BayesOpt VQE Backend finds E = " + std::to_string(r.energy));
	return r;

}

}
}
