import unittest

def allocate_cpu(cpu_usage, vm_capacity):
    if cpu_usage <= 0 or cpu_usage > vm_capacity:
        raise ValueError("Invalid CPU usage")
    return cpu_usage

def run_cloudlet(cpu_usage, vm_capacity):
    if cpu_usage > vm_capacity:
        return "Error: Not enough resources"
    return "Cloudlet running successfully"

def allocate_vm(vm_capacity, required_capacity):
    if required_capacity > vm_capacity:
        return "VM capacity insufficient"
    return "VM allocated successfully"


class TestCPUAllocation(unittest.TestCase):

    def test_valid_allocation(self):
        self.assertEqual(allocate_cpu(50, 100), 50)

    def test_invalid_allocation_high(self):
        with self.assertRaises(ValueError):
            allocate_cpu(150, 100)

    def test_invalid_allocation_low(self):
        with self.assertRaises(ValueError):
            allocate_cpu(0, 100)


class TestCloudletExecution(unittest.TestCase):

    def test_successful_execution(self):
        self.assertEqual(run_cloudlet(50, 100), "Cloudlet running successfully")

    def test_failed_execution(self):
        self.assertEqual(run_cloudlet(150, 100), "Error: Not enough resources")


class TestVMAllocation(unittest.TestCase):

    def test_successful_allocation(self):
        self.assertEqual(allocate_vm(100, 50), "VM allocated successfully")

    def test_failed_allocation(self):
        self.assertEqual(allocate_vm(100, 150), "VM capacity insufficient")


def run_tests():
    # Load the test cases
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCPUAllocation)
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestCloudletExecution))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestVMAllocation))

    # Run the tests
    runner = unittest.TextTestRunner()
    runner.run(suite)


run_tests()


def run_cloudlet(cpu_usage, vm_capacity):
    if cpu_usage > vm_capacity:
        return "Error: Not enough resources"
    return "Cloudlet running successfully"


class TestCloudletExecution(unittest.TestCase):

    def test_successful_execution(self):
        self.assertEqual(run_cloudlet(50, 100), "Cloudlet running successfully")

    def test_failed_execution(self):
        self.assertEqual(run_cloudlet(150, 100), "Error: Not enough resources")


def run_tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCloudletExecution)
    runner = unittest.TextTestRunner()
    runner.run(suite)


run_tests()


# Function for allocating VM
def allocate_vm(vm_capacity, required_capacity):
    if required_capacity > vm_capacity:
        return "VM capacity insufficient"
    return "VM allocated successfully"


# Test case class for VM allocation
class TestVMAllocation(unittest.TestCase):

    def test_successful_allocation(self):
        self.assertEqual(allocate_vm(100, 50), "VM allocated successfully")

    def test_failed_allocation(self):
        self.assertEqual(allocate_vm(100, 150), "VM capacity insufficient")


# Manual test runner
def run_tests():
    # Load tests from the TestVMAllocation class
    suite = unittest.TestLoader().loadTestsFromTestCase(TestVMAllocation)

    # Run the tests
    result = unittest.TextTestRunner().run(suite)

    return result


# Run the tests and capture the result
run_tests()