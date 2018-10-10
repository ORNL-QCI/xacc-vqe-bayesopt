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
	par.crit_name = xacc::optionExists("bo-crit-name") ? std::string(xacc::getOption("bo-crit-name")) : par.crit_name;
	par.kernel.name = xacc::optionExists("bo-kernel-name") ? std::string(xacc::getOption("bo-kernel-name")) : par.kernel.name;
	par.sc_type = xacc::optionExists("bo-score-type") ? str2score(xacc::getOption("bo-score-type").c_str()) : par.sc_type;
	par.surr_name = xacc::optionExists("bo-surrogate-name") ? std::string(xacc::getOption("bo-surrogate-name")) : par.surr_name;
	par.mean.name = xacc::optionExists("bo-mean-name") ? std::string(xacc::getOption("bo-mean-name")) : par.mean.name;
	//par.kernel.hp_mean = xacc::optionExists("bo-kernel-mean") ? std::stod(xacc::getOption("bo-kernel-mean")) : par.kernel.hp_mean;
	//par.kernel.hp_std = xacc::optionExists("bo-kernel-std") ? std::stod(xacc::getOption("bo-kernel-std")) : par.kernel.hp_std;
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

	return r;

}

}
}
