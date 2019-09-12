class Person
  def initialize(name, age)
    @name = name
    @age = age
  end
end

class Student < Person
  def introduce
    print "私の名前は#{self.@name}です。#{self.@age}歳です"
  end
end

student = Student.new("Tom", 20)
student.introduce